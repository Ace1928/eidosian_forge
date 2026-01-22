from PySide2.QtSql import QSqlDatabase, QSqlError, QSqlQuery
from datetime import date

    init_db()
    Initializes the database.
    If tables "books" and "authors" are already in the database, do nothing.
    Return value: None or raises ValueError
    The error value is the QtSql error instance.
    