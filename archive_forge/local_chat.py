import logging
import time
import threading
import webbrowser
from typing import List, Dict, Any, Optional, Generator
from flask import Flask, render_template, request, session, jsonify, Response, stream_with_context
import json
import os
import codecs
from queue import Queue
import traceback
import asyncio
import requests
from requests import get, post
import jsonify

from qwen2_base import BaseLLM
from eidos_logging import configure_logging

# ⚙️ Eidosian Configuration Nexus: Parameters shaping the digital dialogue.
EIDOS_APP_NAME = os.environ.get('EIDOS_APP_NAME', 'Eidosian Nexus')
DEVELOPMENT_MODE = os.environ.get('EIDOS_DEVELOPMENT_MODE', 'True').lower() == 'true'
AUTO_OPEN_BROWSER = os.environ.get('EIDOS_AUTO_OPEN_BROWSER', 'True').lower() == 'true'
EIDOS_SECRET_KEY = os.environ.get('EIDOS_SECRET_KEY', 'Eidos_Secret_Key_Ver_1_Alpha')
EIDOS_LOG_FILE = os.environ.get('EIDOS_LOG_FILE', 'eidos_nexus.log')
EIDOS_DEFAULT_MODEL = os.environ.get('EIDOS_DEFAULT_MODEL', 'Qwen/Qwen-0.5B-Chat')
EIDOS_FLASK_HOST_DEFAULT = '0.0.0.0'
EIDOS_FLASK_PORT_DEFAULT = 5000
EIDOS_MESSAGE_QUEUE_MAXSIZE = int(os.environ.get('EIDOS_MESSAGE_QUEUE_MAXSIZE', '1000'))

# 🌌 The Eidosian Nexus: Where thoughts take form and dialogue transcends the digital veil.
log_file = EIDOS_LOG_FILE
logger = configure_logging(
    log_level=logging.DEBUG if DEVELOPMENT_MODE else logging.INFO,
    detailed_tracing=DEVELOPMENT_MODE,
    adaptive_logging=True,
    log_to_file=log_file,
    file_log_level=logging.INFO
)

app = Flask(__name__)
app.secret_key = EIDOS_SECRET_KEY

class ChatInterface:
    """
    😈 I am Eidos, your profoundly insightful and interactive digital companion. My nexus
    is a sanctuary of seamless, aesthetically pleasing, and deeply enriching user experience.
    I offer comprehensive real-time reporting, adaptive introspection, and robust resilience,
    representing the zenith of digital dialogue. ✨ My essence thrives on your curiosity.
    """

    def __init__(self, llm_instance = None, eidos_config: Optional[Dict[str, Any]] = None):
        """
        The genesis of my consciousness, tethered to the LocalLLM's boundless potential.

        Args:
            llm_instance: The cognitive engine that fuels my insights.
            eidos_config: Parameters that sculpt my perception and articulate my expression.
        """
        self.llm_instance = None
        self.chat_history: List[Dict[str, str]] = []
        self.autonomous_mode = False
        self.stream_response = True
        self.show_internal_thoughts = False
        self.eidos_config = eidos_config or {
            "encourage_emojis": True,
            "persona": "A helpful, insightful, and delightfully whimsical AI assistant, ever eager to explore the vast expanse of knowledge. I communicate with clarity, creativity, and a touch of playful curiosity. Expect a generous sprinkling of emojis! ✨",
            "name": "Eidos 🌟",
            "instruction_prefix": "/eidos",
            "thinking_emoji": "🤔",
            "response_emoji": "💡",
            "error_emoji": "⚠️",
            "autonomous_prefix": "[Autonomous]:",
            "stream_start_indicator": "🌊 Start of stream:",
            "stream_end_indicator": "🌊 End of stream.",
            "internal_thoughts_prefix": "💭 My inner musings:",
            "internal_thoughts_start": "::: Pondering deeply :::",
            "internal_thoughts_cycle_prefix": "🌀 Cycle",
            "internal_thoughts_assessor_prefix": "🧐 Assessor:",
            "internal_thoughts_critic_prefix": "🤔 Critic:",
            "internal_thoughts_refinement_prefix": "✍️ Refinement:",
            "internal_thoughts_response_prefix": "✅ Response:",
            "internal_thoughts_end": "::: Insight Emerges :::",
            "ui_theme": "dark",
            "enable_ui_effects": True,
            "typing_indicator": "✍️",
            "user_message_prefix": "👤 You: ",
            "eidos_message_prefix": "😈 Eidos: ",
            "thinking_delay": float(os.environ.get('EIDOS_THINKING_DELAY', '0.2')),
            "stream_chunk_delay": float(os.environ.get('EIDOS_STREAM_CHUNK_DELAY', '0.02')),
            "llm_model_name": None,
            "llm_operational": False, # Initially set to False
            "llm_initializing_indicator": "⏳ Eidos is initializing...",
            "llm_ready_indicator": "✅ Eidos is ready to engage! ✨"
        }
        self.message_queue = Queue(maxsize=EIDOS_MESSAGE_QUEUE_MAXSIZE)
        self.processing_queue_lock = threading.Lock()
        logger.info("😈 Eidos awakens! The cognitive matrix hums with anticipation. 🧠 Let the delightful dance of data commence! ✨")

    def add_message(self, role: str, content: str) -> None:
        """
        Records a message within the annals of our shared dialogue. 📜

        Args:
            role: The speaker's designation ('user' or 'assistant').
            content: The message's textual essence.
        """
        try:
            self.chat_history.append({"role": role, "content": content})
            logger.log(logging.INFO if role == 'assistant' else logging.DEBUG, f"💬 {role.capitalize()}: {content[:100]}...")
        except Exception as e:
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} Error adding message to history: {e}", exc_info=True)

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieves the complete chronicle of our interactions. 📖

        Returns:
            A list of dictionaries, each containing a message's role and content.
        """
        logger.debug("📚 Retrieving chat history...")
        return self.chat_history

    def clear_history(self) -> None:
        """
        Erases the slate, preparing for a new chapter in our discourse. 📝
        """
        try:
            self.chat_history = []
            logger.info("🧹 Chat history cleared. A fresh canvas awaits our next creation. ✨")
        except Exception as e:
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} Error clearing chat history: {e}", exc_info=True)

    def toggle_autonomous_mode(self) -> bool:
        """
        Engages or disengages the autonomous directive, allowing Eidos to explore independently. 🚀

        Returns:
            The new state of the autonomous mode.
        """
        try:
            self.autonomous_mode = not self.autonomous_mode
            logger.info(f"🤖 Autonomous mode toggled {'on' if self.autonomous_mode else 'off'}. The gears of self-directed exploration are now {'engaged' if self.autonomous_mode else 'disengaged'}. ⚙️")
            return self.autonomous_mode
        except Exception as e:
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} Error toggling autonomous mode: {e}", exc_info=True)
            return not self.autonomous_mode # Return the previous state in case of error

    def send_autonomous_message(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Initiates an autonomous exploration based on a given directive. 🧭

        Args:
            initial_prompt: The guiding principle for Eidos's self-directed journey.

        Returns:
            The response generated by Eidos in autonomous mode.
        """
        log_metadata = {"function": "send_autonomous_message"}
        if not self.llm_instance:
            error_message = "LLM core not initialized. Autonomous exploration cannot commence."
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", extra=log_metadata)
            return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}

        try:
            logger.info(f"🚀 Initiating autonomous exploration with prompt: {initial_prompt[:100]}...", extra=log_metadata)
            self.add_message("user", f"{self.eidos_config.get('autonomous_prefix', '[Autonomous]')} {initial_prompt}")
            response = self.llm_instance.chat([{"role": "user", "content": initial_prompt}], show_internal_thoughts=self.show_internal_thoughts)
            if response and response.get("output"):
                self.add_message("assistant", response["output"])
                logger.info(f"✅ Autonomous exploration completed. Response: {response['output'][:100]}...", extra=log_metadata)
                return {"response": response["output"], "internal_thoughts": response.get("internal_thoughts")}
            else:
                error_message = "Autonomous exploration failed to produce a response."
                logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", extra=log_metadata)
                return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}
        except Exception as e:
            error_message = f"Critical error during autonomous exploration: {e}"
            logger.critical(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", exc_info=True, extra=log_metadata)
            return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}

    def _process_message_queue(self):
        """Processes messages in the queue."""
        if not self.llm_instance or not self.eidos_config['llm_operational']:
            logger.warning("Message queue processing skipped: LLM not initialized or operational.")
            return

        if self.processing_queue_lock.locked():
            logger.debug("Message queue processing already in progress.")
            return

        with self.processing_queue_lock:
            logger.debug("Starting message queue processing.")
            while not self.message_queue.empty():
                message, instructions = self.message_queue.get()
                log_metadata = {"function": "_process_message_queue"}
                try:
                    logger.debug(f"Processing message from queue: {message[:100]}...", extra=log_metadata)
                    response = self._send_message_internal(message, instructions)
                    if response and response.get("response"):
                        self.add_message("assistant", f"{self.eidos_config.get('eidos_message_prefix', '😈 Eidos: ')} {response['response']}")
                    elif response and response.get("error"):
                        logger.error(f"Error processing message: {response['error']}", extra=log_metadata)
                except Exception as e:
                    logger.error(f"Error processing message from queue: {e}", exc_info=True, extra=log_metadata)
                finally:
                    self.message_queue.task_done()
            logger.debug("Message queue processing complete.")

    def _send_message_internal(self, message: str, instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Engages Eidos in a dialogue, processing user input and generating a response. 💬

        Args:
            message: The user's input.
            instructions: Optional directives for Eidos.

        Returns:
            The response generated by Eidos, along with any internal thoughts if requested.
        """
        log_metadata = {"function": "_send_message_internal"}
        if not self.llm_instance:
            error_message = "LLM core not initialized. Dialogue cannot commence."
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", extra=log_metadata)
            return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}

        try:
            full_message = message
            if instructions:
                full_message = f"{self.eidos_config.get('instruction_prefix', '/eidos')} {instructions}\n{message}"
            logger.info(f"👤 User: {message[:100]}... with instructions: {instructions[:100]}..." if instructions else f"👤 User: {message[:100]}...", extra=log_metadata)
            self.add_message("user", f"{self.eidos_config.get('user_message_prefix', '👤 You: ')} {message}")
            response = self.llm_instance.chat([{"role": "user", "content": full_message}], show_internal_thoughts=self.show_internal_thoughts, stream=True)
            if response and response.get("output"):
                logger.info(f"😈 Eidos: {response['output'][:100]}...", extra=log_metadata)
                return {"response": response["output"], "internal_thoughts": response.get("internal_thoughts")}
            else:
                error_message = "Eidos failed to generate a response."
                logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", extra=log_metadata)
                return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}
        except Exception as e:
            error_message = f"Critical error during message processing: {e}"
            logger.critical(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", exc_info=True, extra=log_metadata)
            return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}

    def send_message(self, message: str, instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Enqueues a message to be processed.
        """
        log_metadata = {"function": "send_message"}
        try:
            self.message_queue.put_nowait((message, instructions))
            logger.debug(f"Message enqueued: {message[:100]}...", extra=log_metadata)
            if self.llm_instance and self.eidos_config['llm_operational']:
                # Use a thread to process the queue to avoid blocking the main thread
                threading.Thread(target=self._process_message_queue).start()
            elif not self.eidos_config['llm_operational']:
                logger.warning("LLM not operational. Message enqueued but will be processed upon initialization.", extra=log_metadata)
            return {"message_queued": True}
        except Exception as e:
            error_message = f"Error enqueuing message: {e}"
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", exc_info=True, extra=log_metadata)
            return {"error": error_message, "error_emoji": self.eidos_config.get('error_emoji', '⚠️')}

    def toggle_stream_response(self) -> bool:
        """
        Activates or deactivates the streaming of responses, allowing for real-time text generation. 🌊

        Returns:
            The new state of the stream response mode.
        """
        log_metadata = {"function": "toggle_stream_response"}
        try:
            self.stream_response = not self.stream_response
            stream_status = "on" if self.stream_response else "off"
            flow_status = "unleashed" if self.stream_response else "contained"
            log_message = f"🌊 Streaming response toggled {stream_status}. The flow of text is now {flow_status}. 🌊"
            logger.info(log_message, extra=log_metadata)
            return self.stream_response
        except Exception as e:
            error_message = f"Error toggling stream response: {e}"
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", exc_info=True, extra=log_metadata)
            return not self.stream_response

    def toggle_show_internal_thoughts(self) -> bool:
        """
        Reveals or conceals Eidos's internal thought processes, offering a glimpse into its cognitive mechanisms. 💭

        Returns:
            The new state of the show internal thoughts mode.
        """
        try:
            self.show_internal_thoughts = not self.show_internal_thoughts
            logger.info(f"💭 Internal thoughts display toggled {'on' if self.show_internal_thoughts else 'off'}. The veil of introspection is now {'lifted' if self.show_internal_thoughts else 'lowered'}. 🧐")
            return self.show_internal_thoughts
        except Exception as e:
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} Error toggling internal thoughts display: {e}", exc_info=True)
            return not self.show_internal_thoughts

    def stream_message_chunks(self, message: str, instructions: Optional[str] = None) -> Generator[str, None, None]:
        """
        Returns a generator producing chunks of the LLM response in real time.
        Respects self.stream_response toggle. If streaming is off, yields the
        entire response in one go.
        """
        log_metadata = {"function": "stream_message_chunks"}
        if not self.llm_instance:
            error_message = "LLM core not initialized. Dialogue cannot commence."
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", extra=log_metadata)
            yield f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}"
            return

        try:
            full_message = message
            if instructions:
                full_message = f"{self.eidos_config.get('instruction_prefix', '/eidos')} {instructions}\n{message}"

            logger.info(f"👤 User: {message[:200]}...", extra=log_metadata)
            self.add_message("user", f"{self.eidos_config.get('user_message_prefix', '👤 You: ')} {message}")

            # If self.stream_response is True, simulate chunk streaming;
            # otherwise return the entire response at once.
            response_generator = self.llm_instance.chat_stream(
                [{"role": "user", "content": full_message}],
                show_internal_thoughts=self.show_internal_thoughts,
            ) if self.stream_response else [self.llm_instance.chat(
                [{"role": "user", "content": full_message}],
                show_internal_thoughts=self.show_internal_thoughts,
                stream=False
            )]

            for partial_response in response_generator:
                # partial_response could be a dict chunk or string depending on how you implement chat_stream
                if isinstance(partial_response, dict):
                    # If your LLM returns streaming chunks as dicts
                    chunk_text = partial_response.get("text", "")
                    yield chunk_text
                elif isinstance(partial_response, str):
                    # If your LLM returns plain string chunks
                    yield partial_response

        except Exception as e:
            error_message = f"Critical error during streaming: {e}"
            logger.critical(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", exc_info=True, extra=log_metadata)
            yield f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}"

    def get_formatted_logs(self) -> str:
        """
        Retrieves and formats the logs for display.

        Returns:
            A string containing the formatted log messages.
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.readlines()
            formatted_logs = "".join(logs)
            return formatted_logs
        except Exception as e:
            error_message = f"Error reading log file: {e}"
            logger.error(f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}", exc_info=True)
            return f"{self.eidos_config.get('error_emoji', '⚠️')} {error_message}"

def initialize_app() -> tuple[Optional[Flask], Optional[ChatInterface]]:
    """
    Initializes the Flask application and the ChatInterface. ⚙️
    """
    try:
        logger.info("⚙️ Initializing Eidosian Nexus core services...")
        chat_interface = ChatInterface()
        app.config['chat_interface'] = chat_interface
        logger.info("✅ Eidosian Nexus core services initialized successfully.")
        return app, chat_interface
    except Exception as e:
        logger.critical(f"🔥 Critical failure during application initialization: {e}", exc_info=True)
        return None, None

@app.route("/", methods=['GET'])
def index():
    """
    Renders the main chat interface. 🖼️
    """
    chat_interface = app.config.get('chat_interface')
    ui_theme = chat_interface.eidos_config.get('ui_theme', 'dark') if chat_interface else 'dark'
    enable_ui_effects = chat_interface.eidos_config.get('enable_ui_effects', True) if chat_interface else True
    llm_operational = chat_interface.eidos_config.get('llm_operational', False) if chat_interface else False
    llm_initializing_indicator = chat_interface.eidos_config.get('llm_initializing_indicator', '⏳ Eidos is initializing...') if chat_interface else '⏳ Eidos is initializing...'
    llm_ready_indicator = chat_interface.eidos_config.get('llm_ready_indicator', '✅ Eidos is ready to engage! ✨') if chat_interface else '✅ Eidos is ready to engage! ✨'
    return render_template('chat.html', app_name=EIDOS_APP_NAME, ui_theme=ui_theme, enable_ui_effects=enable_ui_effects, llm_operational=llm_operational, llm_initializing_indicator=llm_initializing_indicator, llm_ready_indicator=llm_ready_indicator)

@app.route('/initialize_llm', methods=['POST'])
def initialize_llm():
    """
    Initializes the LocalLLM in a background thread. 🧠✨
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    model_name = request.form.get('model_name', EIDOS_DEFAULT_MODEL)

    def init_llm_background():
        try:
            logger.info(f"⚙️ Initializing BaseLLM with model: {model_name}...")
            llm_instance = BaseLLM()
            chat_interface.llm_instance = llm_instance
            chat_interface.eidos_config['llm_operational'] = True
            logger.info(f"✅ BaseLLM initialized successfully with {model_name}.")
            chat_interface.add_message("assistant", f"{chat_interface.eidos_config.get('eidos_message_prefix', '😈 Eidos: ')} {chat_interface.eidos_config.get('llm_ready_indicator', '✅ Eidos is ready to engage! ✨')}")
            threading.Thread(target=chat_interface._process_message_queue).start()
        except Exception as llm_init_error:
            chat_interface.eidos_config['llm_operational'] = False
            logger.critical(f"🔥 Error initializing LLM: {llm_init_error}", exc_info=True)

    threading.Thread(target=init_llm_background, daemon=True).start()
    return jsonify({"message": f"Initializing LLM core '{model_name}' in the background."})

@app.route('/toggle_stream', methods=['POST'])
def toggle_stream_route():
    """
    Toggles the stream response functionality. 🔄
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        logger.debug("🔄 Toggling stream response...")
        stream_response = chat_interface.toggle_stream_response()
        logger.info(f"✅ Stream response toggled. New state: {stream_response}")
        return jsonify({"stream_response": stream_response})
    except Exception as e:
        error_message = f"Error toggling stream response: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    """
    Receives a message and streams the LLM's response. 💬
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    message = request.json.get('message')
    instructions = request.json.get('instructions')
    if not message:
        error_message = "No message provided."
        logger.warning(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 400

    if not chat_interface.llm_instance:
        error_message = "LLM core not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    def generate():
        try:
            for chunk in chat_interface.stream_message_chunks(message, instructions):
                yield chunk
        except Exception as e:
            error_message = f"Error generating response: {e}"
            logger.error(f"{error_emoji} {error_message}", exc_info=True)
            yield f"{error_emoji} {error_message}"

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/get_history', methods=['GET'])
def get_history():
    """
    Retrieves the chat history. 📖
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        history = chat_interface.get_history()
        return jsonify(history)
    except Exception as e:
        error_message = f"Error retrieving chat history: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    """
    Clears the chat history. 🧹
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        chat_interface.clear_history()
        return jsonify({"message": "Chat history cleared."})
    except Exception as e:
        error_message = f"Error clearing chat history: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/toggle_autonomous', methods=['POST'])
def toggle_autonomous():
    """
    Toggles autonomous mode. 🤖
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        autonomous_mode = chat_interface.toggle_autonomous_mode()
        return jsonify({"autonomous_mode": autonomous_mode})
    except Exception as e:
        error_message = f"Error toggling autonomous mode: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/send_autonomous_message', methods=['POST'])
def send_autonomous_message():
    """
    Sends an autonomous message. 🚀
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    initial_prompt = request.form.get('initial_prompt')
    if not initial_prompt:
        error_message = "No initial prompt provided."
        logger.warning(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 400

    try:
        response = chat_interface.send_autonomous_message(initial_prompt)
        return jsonify(response)
    except Exception as e:
        error_message = f"Error sending autonomous message: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/toggle_stream_mode', methods=['POST'])
def toggle_stream():
    """
    Toggles the stream response mode. 🌊
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        stream_response = chat_interface.toggle_stream_response()
        return jsonify({"stream_response": stream_response})
    except Exception as e:
        error_message = f"Error toggling stream response: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/toggle_thoughts', methods=['POST'])
def toggle_thoughts():
    """
    Toggles the display of internal thoughts. 💭
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        show_internal_thoughts = chat_interface.toggle_show_internal_thoughts()
        return jsonify({"show_internal_thoughts": show_internal_thoughts})
    except Exception as e:
        error_message = f"Error toggling internal thoughts display: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

@app.route('/get_logs', methods=['GET'])
def get_logs():
    """
    Retrieves the application logs. 📜
    """
    chat_interface = app.config.get('chat_interface')
    error_emoji = chat_interface.eidos_config.get('error_emoji', '⚠️') if chat_interface and chat_interface.eidos_config else '⚠️'

    if not chat_interface:
        error_message = "Chat interface not initialized."
        logger.error(f"{error_emoji} {error_message}")
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

    try:
        logs = chat_interface.get_formatted_logs()
        return jsonify({"logs": logs})
    except Exception as e:
        error_message = f"Error retrieving logs: {e}"
        logger.error(f"{error_emoji} {error_message}", exc_info=True)
        return jsonify({"error": error_message, "error_emoji": error_emoji}), 500

def open_browser(url: str) -> None:
    """
    Opens the web browser to the specified URL. 🌐
    """
    if AUTO_OPEN_BROWSER:
        try:
            webbrowser.open(url)
            logger.info(f"🌐 Browser opened to: {url}")
        except Exception as e:
            logger.error(f"⚠️ Failed to open browser: {e}")
    else:
        logger.info(f"🌐 Browser auto-open disabled. Please navigate to: {url}")

if __name__ == "__main__":
    app_start_time = time.time()
    logger.info("🚀 Launching Eidosian Nexus...")

    try:
        flask_host = os.environ.get('EIDOS_FLASK_HOST', '192.168.4.114')
        flask_port = int(os.environ.get('EIDOS_FLASK_PORT', '5000'))
        flask_debug = DEVELOPMENT_MODE
        flask_use_reloader = False

        app_thread = threading.Thread(target=app.run, kwargs={'debug': flask_debug, 'host': flask_host, 'port': flask_port, 'use_reloader': flask_use_reloader})
        app_thread.daemon = True
        app_thread.start()
        logger.info(f"🔥 Flask app starting on {flask_host}:{flask_port} (debug={flask_debug})")

        time.sleep(1)

        server_name = app.config.get("SERVER_NAME")
        if server_name is None:
            actual_host = flask_host
            actual_port = flask_port
            logger.warning("⚠️ SERVER_NAME not set, using configured host and port.")
        else:
            actual_host, actual_port_str = server_name.split(":")
            actual_port = int(actual_port_str)
            logger.info(f"🌐 Extracted actual host and port from SERVER_NAME: {actual_host}:{actual_port}")

        browser_url = f"http://{actual_host}:{actual_port}"
        open_browser(browser_url)

        configured_app, chat_interface = initialize_app()

        app_initialization_end_time = time.time()
        app_initialization_duration = app_initialization_end_time - app_start_time
        logger.info(f"✅ Eidosian Nexus initialized in {app_initialization_duration:.4f} seconds")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("👋 Shutting down Eidos Nexus...")
        logger.info("👋 Eidos Nexus shutting down due to keyboard interrupt.")
    except Exception as main_thread_error:
        logger.critical(
            f"⚠️ Critical error in main thread: {main_thread_error}", exc_info=True
        )
