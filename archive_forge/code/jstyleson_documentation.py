import json
Clear all comments in json_str.

    Clear JS-style comments like // and /**/ in json_str.
    Accept a str or unicode as input.

    Args:
        json_str: A json string of str or unicode to clean up comment

    Returns:
        str: The str without comments (or unicode if you pass in unicode)
    