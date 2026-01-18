import datetime
from redis.utils import str_if_bytes

    Handle SET result since GET argument is available since Redis 6.2.
    Parsing SET result into:
    - BOOL
    - String when GET argument is used
    