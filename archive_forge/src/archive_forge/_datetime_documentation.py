from datetime import datetime, timezone

    Parses a date_string returned from the server to a datetime object.

    This parser is a weak-parser is the sense that it handles only a single format of
    date_string. It is expected that the server format will never change. The
    implementation depends only on the standard lib to avoid an external dependency
    (python-dateutil). See full discussion about this decision on PR:
    https://github.com/huggingface/huggingface_hub/pull/999.

    Example:
        ```py
        > parse_datetime('2022-08-19T07:19:38.123Z')
        datetime.datetime(2022, 8, 19, 7, 19, 38, 123000, tzinfo=timezone.utc)
        ```

    Args:
        date_string (`str`):
            A string representing a datetime returned by the Hub server.
            String is expected to follow '%Y-%m-%dT%H:%M:%S.%fZ' pattern.

    Returns:
        A python datetime object.

    Raises:
        :class:`ValueError`:
            If `date_string` cannot be parsed.
    