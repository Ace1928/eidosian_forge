def _iterdump(connection):
    """
    Returns an iterator to the dump of the database in an SQL text format.

    Used to produce an SQL dump of the database.  Useful to save an in-memory
    database for later restoration.  This function should not be called
    directly but instead called from the Connection method, iterdump().
    """
    writeable_schema = False
    cu = connection.cursor()
    yield 'BEGIN TRANSACTION;'
    q = '\n        SELECT "name", "type", "sql"\n        FROM "sqlite_master"\n            WHERE "sql" NOT NULL AND\n            "type" == \'table\'\n            ORDER BY "name"\n        '
    schema_res = cu.execute(q)
    sqlite_sequence = []
    for table_name, type, sql in schema_res.fetchall():
        if table_name == 'sqlite_sequence':
            rows = cu.execute('SELECT * FROM "sqlite_sequence";').fetchall()
            sqlite_sequence = ['DELETE FROM "sqlite_sequence"']
            sqlite_sequence += [f"""INSERT INTO "sqlite_sequence" VALUES('{row[0]}',{row[1]})""" for row in rows]
            continue
        elif table_name == 'sqlite_stat1':
            yield 'ANALYZE "sqlite_master";'
        elif table_name.startswith('sqlite_'):
            continue
        elif sql.startswith('CREATE VIRTUAL TABLE'):
            if not writeable_schema:
                writeable_schema = True
                yield 'PRAGMA writable_schema=ON;'
            yield "INSERT INTO sqlite_master(type,name,tbl_name,rootpage,sql)VALUES('table','{0}','{0}',0,'{1}');".format(table_name.replace("'", "''"), sql.replace("'", "''"))
        else:
            yield '{0};'.format(sql)
        table_name_ident = table_name.replace('"', '""')
        res = cu.execute('PRAGMA table_info("{0}")'.format(table_name_ident))
        column_names = [str(table_info[1]) for table_info in res.fetchall()]
        q = 'SELECT \'INSERT INTO "{0}" VALUES({1})\' FROM "{0}";'.format(table_name_ident, ','.join(('\'||quote("{0}")||\''.format(col.replace('"', '""')) for col in column_names)))
        query_res = cu.execute(q)
        for row in query_res:
            yield '{0};'.format(row[0])
    q = '\n        SELECT "name", "type", "sql"\n        FROM "sqlite_master"\n            WHERE "sql" NOT NULL AND\n            "type" IN (\'index\', \'trigger\', \'view\')\n        '
    schema_res = cu.execute(q)
    for name, type, sql in schema_res.fetchall():
        yield '{0};'.format(sql)
    if writeable_schema:
        yield 'PRAGMA writable_schema=OFF;'
    for row in sqlite_sequence:
        yield '{0};'.format(row)
    yield 'COMMIT;'