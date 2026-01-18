def decode_message(input_file):
    import math

    # Read and parse the input file
    number_word_map = {}
    with open(input_file, "r") as file:
        for line in file:
            parts = line.split()
            number = int(parts[0])
            word = " ".join(parts[1:])  # Joining in case there are multi-word entries
            number_word_map[number] = word

    # Determine the maximum number to figure out the pyramid size
    max_number = max(number_word_map.keys())

    # Create a list to store the last numbers on each level of the pyramid
    last_numbers = []

    # Fill the pyramid levels
    current_number = 1
    level = 1
    while current_number <= max_number:
        # Calculate the last number for this level
        last_number = current_number + level - 1
        if last_number <= max_number:
            last_numbers.append(last_number)
        current_number = last_number + 1
        level += 1

    # Collect the words that correspond to the last numbers of each pyramid level
    words = [number_word_map[num] for num in last_numbers if num in number_word_map]

    # Form the message from the collected words
    message = " ".join(words)
    return message


# Example usage
file_path = "/home/lloyd/Downloads/coding_qual_input.txt"
message = decode_message(file_path)
print("Decoded message:", message)
