from django.db.models.expressions import OrderByList
def as_sql(self, compiler, connection):
    order_by_sql, order_by_params = compiler.compile(self.order_by)
    sql, sql_params = super().as_sql(compiler, connection, ordering=order_by_sql)
    return (sql, (*sql_params, *order_by_params))