from __future__ import print_function, absolute_import
from PySide2.QtWidgets import (QAction, QAbstractItemView, qApp, QDataWidgetMapper,
from PySide2.QtGui import QKeySequence
from PySide2.QtSql import (QSqlRelation, QSqlRelationalTableModel, QSqlTableModel,
from PySide2.QtCore import QAbstractItemModel, QObject, QSize, Qt, Slot
import createdb
from ui_bookwindow import Ui_BookWindow
from bookdelegate import BookDelegate
class BookWindow(QMainWindow, Ui_BookWindow):
    """A window to show the books available"""

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        createdb.init_db()
        model = QSqlRelationalTableModel(self.bookTable)
        model.setEditStrategy(QSqlTableModel.OnManualSubmit)
        model.setTable('books')
        author_idx = model.fieldIndex('author')
        genre_idx = model.fieldIndex('genre')
        model.setRelation(author_idx, QSqlRelation('authors', 'id', 'name'))
        model.setRelation(genre_idx, QSqlRelation('genres', 'id', 'name'))
        model.setHeaderData(author_idx, Qt.Horizontal, self.tr('Author Name'))
        model.setHeaderData(genre_idx, Qt.Horizontal, self.tr('Genre'))
        model.setHeaderData(model.fieldIndex('title'), Qt.Horizontal, self.tr('Title'))
        model.setHeaderData(model.fieldIndex('year'), Qt.Horizontal, self.tr('Year'))
        model.setHeaderData(model.fieldIndex('rating'), Qt.Horizontal, self.tr('Rating'))
        if not model.select():
            print(model.lastError())
        self.bookTable.setModel(model)
        self.bookTable.setItemDelegate(BookDelegate(self.bookTable))
        self.bookTable.setColumnHidden(model.fieldIndex('id'), True)
        self.bookTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.authorEdit.setModel(model.relationModel(author_idx))
        self.authorEdit.setModelColumn(model.relationModel(author_idx).fieldIndex('name'))
        self.genreEdit.setModel(model.relationModel(genre_idx))
        self.genreEdit.setModelColumn(model.relationModel(genre_idx).fieldIndex('name'))
        self.bookTable.horizontalHeader().setSectionResizeMode(model.fieldIndex('rating'), QHeaderView.ResizeToContents)
        mapper = QDataWidgetMapper(self)
        mapper.setModel(model)
        mapper.setItemDelegate(BookDelegate(self))
        mapper.addMapping(self.titleEdit, model.fieldIndex('title'))
        mapper.addMapping(self.yearEdit, model.fieldIndex('year'))
        mapper.addMapping(self.authorEdit, author_idx)
        mapper.addMapping(self.genreEdit, genre_idx)
        mapper.addMapping(self.ratingEdit, model.fieldIndex('rating'))
        selection_model = self.bookTable.selectionModel()
        selection_model.currentRowChanged.connect(mapper.setCurrentModelIndex)
        self.bookTable.setCurrentIndex(model.index(0, 0))
        self.create_menubar()

    def showError(err):
        QMessageBox.critical(self, 'Unable to initialize Database', 'Error initializing database: ' + err.text())

    def create_menubar(self):
        file_menu = self.menuBar().addMenu(self.tr('&File'))
        quit_action = file_menu.addAction(self.tr('&Quit'))
        quit_action.triggered.connect(qApp.quit)
        help_menu = self.menuBar().addMenu(self.tr('&Help'))
        about_action = help_menu.addAction(self.tr('&About'))
        about_action.setShortcut(QKeySequence.HelpContents)
        about_action.triggered.connect(self.about)
        aboutQt_action = help_menu.addAction('&About Qt')
        aboutQt_action.triggered.connect(qApp.aboutQt)

    def about(self):
        QMessageBox.about(self, self.tr('About Books'), self.tr('<p>The <b>Books</b> example shows how to use Qt SQL classes with a model/view framework.'))