from functools import reduce
def apply_functional_programming_techniques(data):
    """
    Applies functional programming techniques to process data.

    Args:
    data (list): The data to be processed.

    Returns:
    tuple: A tuple containing the results of map, filter, and reduce operations.
    """
    mapped_data = list(map(lambda x: x * 2, data))
    filtered_data = list(filter(lambda x: x % 2 == 0, data))
    reduced_data = reduce(lambda x, y: x + y, data)
    return (mapped_data, filtered_data, reduced_data)