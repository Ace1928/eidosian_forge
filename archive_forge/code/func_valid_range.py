import logging
import re
def valid_range(range_, loose):
    try:
        return make_range(range_, loose).range or '*'
    except Exception:
        return None