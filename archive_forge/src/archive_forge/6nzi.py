import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QListWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt


class GUI(QMainWindow):
    """
    Manages the graphical user interface, displaying the list of processed files and providing interactive functionalities.
    This implementation uses PyQt5 to create a robust, cross-platform GUI.
    """

    def __init__(self):
        """
        Initializes the GUI components.
        """
        super().__init__()
        self.setWindowTitle("Text and Embedding Analysis Tool")
        self.setGeometry(100, 100, 800, 600)  # Position and size: x, y, width, height

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout()
        self.main_widget.setLayout(self.layout)

        self.file_list_widget = QListWidget()
        self.file_list_widget.clicked.connect(self.on_file_selected)

        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)

        self.layout.addWidget(self.file_list_widget, 1)
        self.layout.addWidget(self.content_display, 2)

    def main_loop(self) -> None:
        """
        Starts the GUI event loop.
        """
        app = QApplication(sys.argv)
        self.show()
        sys.exit(app.exec_())

    def update_file_list(self, file_list: list) -> None:
        """
        Updates the display of processed files.
        :param file_list: list of str - List of file paths to display in the GUI.
        """
        self.file_list_widget.clear()
        self.file_list_widget.addItems(file_list)

    def on_file_selected(self):
        """
        Handles the selection of a file from the list, displaying its contents and related data.
        """
        selected_item = self.file_list_widget.currentItem()
        file_path = selected_item.text()
        self.show_file_contents(file_path)

    def show_file_contents(self, file_path: str) -> None:
        """
        Displays the contents of a selected file along with related analysis and embeddings in a side pane.
        :param file_path: str - Path to the file whose contents are to be displayed.
        """
        # This function needs to integrate with the backend to fetch the file's content and analysis results
        content = f"Contents and analysis for {file_path}"  # Placeholder for actual content fetching
        self.content_display.setText(content)
