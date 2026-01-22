import base64
import pickle
from datetime import datetime, timezone
from django.conf import settings
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.db import DatabaseError, connections, models, router, transaction
from django.utils.timezone import now as tz_now
def _base_set(self, mode, key, value, timeout=DEFAULT_TIMEOUT):
    timeout = self.get_backend_timeout(timeout)
    db = router.db_for_write(self.cache_model_class)
    connection = connections[db]
    quote_name = connection.ops.quote_name
    table = quote_name(self._table)
    with connection.cursor() as cursor:
        cursor.execute('SELECT COUNT(*) FROM %s' % table)
        num = cursor.fetchone()[0]
        now = tz_now()
        now = now.replace(microsecond=0)
        if timeout is None:
            exp = datetime.max
        else:
            tz = timezone.utc if settings.USE_TZ else None
            exp = datetime.fromtimestamp(timeout, tz=tz)
        exp = exp.replace(microsecond=0)
        if num > self._max_entries:
            self._cull(db, cursor, now, num)
        pickled = pickle.dumps(value, self.pickle_protocol)
        b64encoded = base64.b64encode(pickled).decode('latin1')
        try:
            with transaction.atomic(using=db):
                cursor.execute('SELECT %s, %s FROM %s WHERE %s = %%s' % (quote_name('cache_key'), quote_name('expires'), table, quote_name('cache_key')), [key])
                result = cursor.fetchone()
                if result:
                    current_expires = result[1]
                    expression = models.Expression(output_field=models.DateTimeField())
                    for converter in connection.ops.get_db_converters(expression) + expression.get_db_converters(connection):
                        current_expires = converter(current_expires, expression, connection)
                exp = connection.ops.adapt_datetimefield_value(exp)
                if result and mode == 'touch':
                    cursor.execute('UPDATE %s SET %s = %%s WHERE %s = %%s' % (table, quote_name('expires'), quote_name('cache_key')), [exp, key])
                elif result and (mode == 'set' or (mode == 'add' and current_expires < now)):
                    cursor.execute('UPDATE %s SET %s = %%s, %s = %%s WHERE %s = %%s' % (table, quote_name('value'), quote_name('expires'), quote_name('cache_key')), [b64encoded, exp, key])
                elif mode != 'touch':
                    cursor.execute('INSERT INTO %s (%s, %s, %s) VALUES (%%s, %%s, %%s)' % (table, quote_name('cache_key'), quote_name('value'), quote_name('expires')), [key, b64encoded, exp])
                else:
                    return False
        except DatabaseError:
            return False
        else:
            return True