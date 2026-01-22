from django.db.migrations.serializer import BaseSerializer
class RangeSerializer(BaseSerializer):

    def serialize(self):
        module = self.value.__class__.__module__
        module = 'psycopg2.extras' if module == 'psycopg2._range' else module
        return ('%s.%r' % (module, self.value), {'import %s' % module})