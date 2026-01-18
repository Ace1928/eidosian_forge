import re
def to_snake(camel: str) -> str:
    """Convert a PascalCase or camelCase string to snake_case.

    Args:
        camel: The string to convert.

    Returns:
        The converted string in snake_case.
    """
    snake = re.sub('([A-Z]+)([A-Z][a-z])', lambda m: f'{m.group(1)}_{m.group(2)}', camel)
    snake = re.sub('([a-z])([A-Z])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
    snake = re.sub('([0-9])([A-Z])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
    snake = re.sub('([a-z])([0-9])', lambda m: f'{m.group(1)}_{m.group(2)}', snake)
    return snake.lower()