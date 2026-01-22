import random
import uuid
def _generate_random_name(sep='-', integer_scale=3, max_length=20):
    """Helper function for generating a random predicate, noun, and integer combination

    Args:
        sep: String separator for word spacing.
        integer_scale: Dictates the maximum scale range for random integer sampling (power of 10).
        max_length: Maximum allowable string length.

    Returns:
        A random string phrase comprised of a predicate, noun, and random integer.

    """
    name = None
    for _ in range(10):
        name = _generate_string(sep, integer_scale)
        if len(name) <= max_length:
            return name
    return name[:max_length]