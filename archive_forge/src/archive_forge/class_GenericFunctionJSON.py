from datetime import datetime, date
from decimal import Decimal
from json import JSONEncoder
class GenericFunctionJSON(GenericJSON):

    def default(self, obj):
        return jsonify(obj)