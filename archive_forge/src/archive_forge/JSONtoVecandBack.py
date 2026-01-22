import json
from typing import List, Any

def json_to_vector(json_entry: str) -> List[Any]:
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
