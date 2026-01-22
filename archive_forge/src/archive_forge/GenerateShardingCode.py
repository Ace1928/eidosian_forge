
import hashlib

def generate_sharding_code(standard):
    try:
        # Concatenating all relevant fields into a single string
        combined_string = ''.join(str(standard[key]) for key in standard)
        
        # Hashing the combined string for a unique sharding code
        hash_object = hashlib.sha256(combined_string.encode())
        sharding_code = hash_object.hexdigest()
        
        return sharding_code
    except Exception as e:
        print(f"Error in generate_sharding_code: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # This is a placeholder for a standard item, replace with actual data
    standard_example = {
        'StandardID': 1, 
        'ConcatenatedID': '12345', 
        'RegistryCode': 'abcd1234'
    }

    try:
        sharding_code = generate_sharding_code(standard_example)
        print(f'Sharding Code: {sharding_code}')
    except Exception as e:
        print(f"An error occurred: {e}")
