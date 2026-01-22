import datetime
from json import JSONEncoder
class DatetimeJSONEncoder(JSONEncoder):
    """A JSON encoder that understands datetime objects.

    Datetime objects are formatted according to ISO 1601.
    """

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return JSONEncoder.default(self, obj)