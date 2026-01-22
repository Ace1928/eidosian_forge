import json
import hashlib
import logging
from typing import List, Any, Dict

# Setting up basic configuration for logging
logging.basicConfig(filename='library_management.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

def generate_code(standard: Dict, mode: str) -> str:
    """
    Generate a unique hash code for a given standard.

    Parameters:
    - standard (Dict): The standard dictionary.
    - mode (str): The mode of operation ('registry' or 'sharding').

    Returns:
    - str: The generated hash code.
    """
    try:
        if mode == 'registry':
            unique_str = str(standard)
        elif mode == 'sharding':
            unique_str = ''.join(str(standard[key]) for key in standard)
        else:
            raise ValueError("Invalid mode. Choose either 'registry' or 'sharding'.")
        
        hash_object = hashlib.sha256(unique_str.encode())
        return hash_object.hexdigest()
    except Exception as e:
        logging.error(f"Error in generate_{mode}_code: {e}")
        raise


import json

def load_large_json(file_path):
    with open(file_path, 'r') as file:
        json_string = file.read().strip()
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(json_string):
            # Skip leading whitespace
            while idx < len(json_string) and json_string[idx].isspace():
                idx += 1
            if idx < len(json_string):
                obj, end = decoder.raw_decode(json_string[idx:])
                yield obj
                idx += end

def process_json_file(file_path: str) -> str:
    """
    Process a JSON file and update each standard with a unique ID, RegistryCode, and ShardingCode.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - str: The path to the updated JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        for standard in json_data['Standards']:
            standard['ID'] = generate_code(standard, 'registry')
            standard['RegistryCode'] = generate_code(standard, 'registry')
            standard['ShardingCode'] = generate_code(standard, 'sharding')

        updated_file_path = file_path.replace('.json', '_updated.json')
        with open(updated_file_path, 'w') as file:
            json.dump(json_data, file, indent=4)

        logging.info(f"Successfully processed and updated the JSON file: {updated_file_path}")
        return updated_file_path
    except Exception as e:
        logging.error(f"Error processing the JSON file: {e}")
        raise

def json_to_vector(json_entry: str) -> List[Any]:
    """
    Convert a JSON entry to a vector.

    Parameters:
    - json_entry (str): The JSON entry as a string.

    Returns:
    - List[Any]: The converted vector.
    """
    try:
        # Convert JSON to dictionary
        dict_entry = json.loads(json_entry)
        
        # Convert dictionary to vector
        vector_entry = list(dict_entry.values())
        
        return vector_entry
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def vector_to_json(vector_entry: List[Any], json_template: str) -> str:
    """
    Convert a vector to a JSON entry.

    Parameters:
    - vector_entry (List[Any]): The vector to convert.
    - json_template (str): The JSON template to use for conversion.

    Returns:
    - str: The converted JSON entry.
    """
    try:
        # Convert JSON to dictionary
        dict_template = json.loads(json_template)
        
        # Create a new dictionary with the same keys as the template
        dict_entry = {key: value for key, value in zip(dict_template.keys(), vector_entry)}
        
        # Convert dictionary to JSON
        json_entry = json.dumps(dict_entry)
        
        return json_entry
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python this_script.py <path_to_json_file>")
    else:
        input_file = sys.argv[1]
        try:
            output_file = process_json_file(input_file)
            print(f"Processed file saved as: {output_file}")
        except Exception as e:
            print(f"An error occurred: {e}")
