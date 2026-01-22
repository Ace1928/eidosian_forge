from django.db.models import DateTimeField, Func, UUIDField
class RandomUUID(Func):
    template = 'GEN_RANDOM_UUID()'
    output_field = UUIDField()