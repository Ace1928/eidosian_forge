from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union
import tempfile
import os
from datetime import datetime
import traceback
from pathlib import Path
import io


app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the browser

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chat_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
TEMP_DIR = Path(tempfile.gettempdir()) / "chat_history"
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_AGE_HOURS = 24  # Files older than this will be cleaned up

def cleanup_old_files() -> None:
    """Clean up temporary files older than MAX_FILE_AGE_HOURS."""
    try:
        current_time = datetime.now()
        for file_path in TEMP_DIR.glob("*"):
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > (MAX_FILE_AGE_HOURS * 3600):
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")

def parse_chat_history_text(chat_history_text: str) -> List[Dict[str, Any]]:
    """
    Parses the chat history from a text string and returns a list of message dictionaries.
    """
    messages = []
    current_message: Dict[str, Any] = {}
    section: Optional[str] = None
    in_code_block = False
    current_code = ""
    current_code_language = "plaintext"

    # Define regex patterns
    patterns = {
        'message_start': re.compile(r'^Message\s+(\d+):\s*$'),
        'author': re.compile(r'^Author:\s*(.*)$'),
        'timestamp': re.compile(r'^Timestamp:\s*(.*)$'),
        'text_content': re.compile(r'^Text Content:\s*$'),
        'code_content': re.compile(r'^Code Content:\s*$'),
        'attachments': re.compile(r'^Attachments:\s*$'),
        'code_fence': re.compile(r'^```(\w+)?\s*$'),
        'separator': re.compile(r'^-+\s*$'),
        'image': re.compile(r'^Image:\s*(.*)$'),
        'link': re.compile(r'^Link:\s*\[(.*?)\]\((.*?)\)$'),
        'thought': re.compile(r'^Thought:\s*(.*)$')  # Added pattern for thought indicators
    }

    lines = chat_history_text.split('\n')
    for line_num, line in enumerate(lines, 1):
        try:
            line = line.rstrip('\n')

            # Detect code fences
            code_fence_match = patterns['code_fence'].match(line)
            if code_fence_match:
                if not in_code_block:
                    in_code_block = True
                    current_code_language = code_fence_match.group(1) or "plaintext"
                    current_code = ""
                    logger.debug(f"Start of code block with language: {current_code_language}")
                else:
                    in_code_block = False
                    if 'code_content' not in current_message:
                        current_message['code_content'] = []
                    current_message['code_content'].append({
                        "language": current_code_language,
                        "code": current_code.strip()
                    })
                    logger.debug(f"End of code block with language: {current_code_language}")
                    current_code = ""
                    current_code_language = "plaintext"
                continue

            if in_code_block:
                current_code += line + "\n"
                continue

            # Process message start
            message_start_match = patterns['message_start'].match(line)
            if message_start_match:
                if current_message:
                    messages.append(current_message)
                    logger.debug(f"Appended Message {current_message.get('message_number', 'N/A')}")
                
                message_number = int(message_start_match.group(1))
                current_message = {
                    'message_number': message_number,
                    'message_id': len(messages) + 1,
                    'author': '',
                    'timestamp': '',
                    'text_content': '',
                    'code_content': [],
                    'attachments': [],
                    'thought': '',  # Added field for thought indicators
                    'model': '',    # Added field for model information
                    'metadata': {}  # Added field for additional metadata
                }
                section = None
                logger.debug(f"Started parsing Message {message_number}")
                continue

            # Process other patterns
            for pattern_name, pattern in patterns.items():
                if pattern_name in ['message_start', 'code_fence']:
                    continue
                
                match = pattern.match(line)
                if match:
                    if pattern_name == 'author':
                        current_message['author'] = match.group(1).strip()
                    elif pattern_name == 'timestamp':
                        current_message['timestamp'] = match.group(1).strip()
                    elif pattern_name == 'thought':
                        current_message['thought'] = match.group(1).strip()
                    elif pattern_name in ['text_content', 'code_content', 'attachments']:
                        section = pattern_name
                    elif pattern_name == 'separator':
                        section = None
                    elif pattern_name == 'image':
                        current_message['attachments'].append({
                            "type": "image",
                            "src": match.group(1).strip()
                        })
                    elif pattern_name == 'link':
                        current_message['attachments'].append({
                            "type": "link",
                            "text": match.group(1).strip(),
                            "href": match.group(2).strip()
                        })
                    break
            else:
                # No pattern matched, append content to current section
                if section == 'text_content':
                    current_message['text_content'] += line + "\n"
                elif section == 'attachments' and line.strip():
                    # Handle plain text attachments
                    current_message['attachments'].append({
                        "type": "text",
                        "content": line.strip()
                    })

        except Exception as e:
            logger.error(f"Error processing line {line_num}: {line}\nError: {str(e)}")
            raise

    # Process final message
    if current_message:
        if in_code_block:
            current_message['code_content'].append({
                "language": current_code_language,
                "code": current_code.strip()
            })
        messages.append(current_message)
        logger.debug(f"Appended final Message {current_message.get('message_number', 'N/A')}")

    # Post-processing
    for msg in messages:
        msg['text_content'] = msg['text_content'].strip()
        
    return messages

def validate_structured_chat_history(chat_history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validates and extracts messages from a structured chat history dictionary.
    Returns a list of message dictionaries if valid, otherwise raises an error.
    """
    if not isinstance(chat_history, dict):
        raise ValueError(f"Expected dictionary, got {type(chat_history)}")

    if 'messages' not in chat_history:
        raise ValueError("'messages' key not found in structured chat_history")

    messages = chat_history['messages']
    if not isinstance(messages, list):
        raise ValueError(f"'messages' should be a list, got {type(messages)}")

    required_keys = {'message_number', 'author', 'timestamp', 'text_content', 'code_content', 'attachments'}
    optional_keys = {'thought', 'model', 'metadata'}  # Added optional keys
    
    validated_messages = []
    for idx, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            raise ValueError(f"Message at index {idx} is not a dictionary")
            
        missing_keys = required_keys - set(message.keys())
        if missing_keys:
            raise ValueError(f"Message {idx} is missing required keys: {missing_keys}")

        # Add optional fields if missing
        for key in optional_keys:
            if key not in message:
                message[key] = ''

        # Validate types and structure
        if not isinstance(message['attachments'], list):
            raise ValueError(f"'attachments' in message {idx} must be a list")
        if not isinstance(message['code_content'], list):
            raise ValueError(f"'code_content' in message {idx} must be a list")
            
        # Validate attachment structure
        for att_idx, attachment in enumerate(message['attachments']):
            if not isinstance(attachment, dict) or 'type' not in attachment:
                raise ValueError(f"Invalid attachment format at message {idx}, attachment {att_idx}")

        # Validate code content structure
        for code_idx, code_block in enumerate(message['code_content']):
            if not isinstance(code_block, dict) or 'language' not in code_block or 'code' not in code_block:
                raise ValueError(f"Invalid code block format at message {idx}, block {code_idx}")

        validated_messages.append(message)

    logger.debug(f"Validated {len(validated_messages)} messages successfully")
    return validated_messages

def generate_formatted_output(messages: List[Dict[str, Any]], format: str = 'txt') -> Union[str, bytes]:
    """
    Generate formatted output of the chat history in various formats.
    """
    if format == 'txt':
        output = []
        for msg in messages:
            output.append(f"Message {msg['message_number']}:")
            output.append(f"Author: {msg['author']}")
            output.append(f"Timestamp: {msg['timestamp']}")
            
            if msg.get('model'):  # Added model information
                output.append(f"Model: {msg['model']}")
            
            if msg.get('thought'):  # Added thought indicator
                output.append(f"Thought: {msg['thought']}")
            
            if msg['text_content']:
                output.append("Text Content:")
                output.append(msg['text_content'])
            
            if msg['code_content']:
                output.append("Code Content:")
                for code_block in msg['code_content']:
                    output.append(f"```{code_block['language']}")
                    output.append(code_block['code'])
                    output.append("```")
            
            if msg['attachments']:
                output.append("Attachments:")
                for attachment in msg['attachments']:
                    if attachment['type'] == 'image':
                        output.append(f"Image: {attachment['src']}")
                    elif attachment['type'] == 'link':
                        output.append(f"Link: [{attachment['text']}]({attachment['href']})")
                    elif attachment['type'] == 'text':
                        output.append(attachment['content'])
            
            if msg.get('metadata'):  # Added metadata section
                output.append("Metadata:")
                for key, value in msg['metadata'].items():
                    output.append(f"{key}: {value}")
            
            output.append("-" * 50)
            
        return "\n".join(output)
    
    elif format == 'json':
        return json.dumps({'messages': messages}, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

@app.route('/upload_chat', methods=['POST'])
def upload_chat():
    """
    Endpoint to receive chat history and return structured JSON.
    Accepts both raw text and structured JSON formats.
    """
    cleanup_old_files()  # Cleanup old temporary files
    
    try:
        logger.info("Received upload_chat request")
        if not request.is_json:
            raise ValueError("Request must be JSON")

        data = request.get_json()
        if not data or 'chat_history' not in data:
            raise ValueError("'chat_history' key not found in request data")

        chat_history = data['chat_history']
        format_type = data.get('format', 'json')
        
        if not isinstance(format_type, str) or format_type not in ['json', 'txt']:
            raise ValueError("Invalid format type. Must be 'json' or 'txt'")

        # Process the chat history
        if isinstance(chat_history, str):
            logger.info("Processing raw text chat history")
            messages = parse_chat_history_text(chat_history)
        elif isinstance(chat_history, dict):
            logger.info("Processing structured JSON chat history")
            messages = validate_structured_chat_history(chat_history)
        else:
            raise ValueError(f"Unsupported chat_history type: {type(chat_history)}")

        # Generate formatted output
        output_content = generate_formatted_output(messages, format_type)
        
        # Create temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.{format_type}"
        file_path = TEMP_DIR / filename

        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output_content)

        logger.info(f"Created output file: {file_path}")

        # Return file download response
        return send_file(
            file_path,
            mimetype='application/json' if format_type == 'json' else 'text/plain',
            as_attachment=True,
            download_name=filename
        )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
