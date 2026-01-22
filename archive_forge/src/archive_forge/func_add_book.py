from PySide2.QtSql import QSqlDatabase, QSqlError, QSqlQuery
from datetime import date
def add_book(q, title, year, authorId, genreId, rating):
    q.addBindValue(title)
    q.addBindValue(year)
    q.addBindValue(authorId)
    q.addBindValue(genreId)
    q.addBindValue(rating)
    q.exec_()