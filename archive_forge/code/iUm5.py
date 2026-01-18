import sys
import logging

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QAction,
    QLineEdit,
    QStatusBar,
    QDialog,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtGui import QKeySequence
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomWebEnginePage(QWebEnginePage):
    def certificateError(self, certificateError):
        logging.error(f"SSL Error: {certificateError.errorDescription()}")
        return certificateError.isOverridable()


class BrowserUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        try:
            self.setup_central_widget()
            self.create_navbar()
            self.create_menu()
            self.create_status_bar()
            logging.info("UI initialized successfully.")
            self.showMaximized()
        except Exception as e:
            logging.error(f"Failed to initialize UI: {e}")

    def setup_central_widget(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

    def create_status_bar(self):
        try:
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            logging.info("Status bar created successfully.")
        except Exception as e:
            logging.error(f"Failed to create status bar: {e}")

    def create_navbar(self):
        try:
            self.navbar = QToolBar("Navigation")
            self.addToolBar(self.navbar)
            self.add_navbar_actions()
            logging.info("Navigation bar created successfully.")
        except Exception as e:
            logging.error(f"Failed to create navigation bar: {e}")

    def add_navbar_actions(self):
        actions = [
            ("⬅️ Back", QKeySequence.Back, self.browser.back),
            ("➡️ Forward", QKeySequence.Forward, self.browser.forward),
            ("🔄 Reload", QKeySequence.Refresh, self.browser.reload),
            ("🏠 Home", "Ctrl+H", self.navigate_home),
            ("⛔ Stop", "Esc", self.browser.stop),
            ("🔍 Scrape", "Ctrl+S", self.scrape_page),
        ]
        for name, shortcut, handler in actions:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            self.navbar.addAction(action)

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        self.navbar.addWidget(self.url_bar)

    def create_menu(self):
        try:
            menubar = self.menuBar()
            self.create_file_menu(menubar)
            self.create_view_menu(menubar)
            logging.info("Menu created successfully.")
        except Exception as e:
            logging.error(f"Failed to create menu: {e}")

    def create_file_menu(self, menubar):
        file_menu = menubar.addMenu("&File")
        actions = [
            ("🖨️ Print", QKeySequence.Print, self.print_page),
            ("❌ Exit", QKeySequence.Quit, self.close),
        ]
        for name, shortcut, handler in actions:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            file_menu.addAction(action)

    def create_view_menu(self, menubar):
        view_menu = menubar.addMenu("&View")
        actions = [
            (
                "🔍➕ Zoom In",
                QKeySequence.ZoomIn,
                lambda: self.browser.setZoomFactor(self.browser.zoomFactor() + 0.1),
            ),
            (
                "🔍➖ Zoom Out",
                QKeySequence.ZoomOut,
                lambda: self.browser.setZoomFactor(self.browser.zoomFactor() - 0.1),
            ),
            ("🔄 Reset Zoom", "Ctrl+0", lambda: self.browser.setZoomFactor(1.0)),
        ]
        for name, shortcut, handler in actions:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            view_menu.addAction(action)

    def navigate_home(self):
        try:
            self.browser.setUrl(QUrl("http://www.google.com"))
            logging.info("Navigated to home.")
        except Exception as e:
            logging.error(f"Failed to navigate home: {e}")

    def navigate_to_url(self):
        try:
            url = self.url_bar.text()
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url
            self.browser.setUrl(QUrl(url))
            logging.info(f"Navigated to URL: {url}")
        except Exception as e:
            logging.error(f"Failed to navigate to URL: {e}")

    def print_page(self):
        try:
            printer = QPrinter()
            dialog = QPrintDialog(printer, self)
            if dialog.exec_() == QDialog.Accepted:
                self.browser.page().print(
                    printer,
                    lambda success: self.status.showMessage(
                        "Printing completed" if success else "Printing failed"
                    ),
                )
                logging.info("Print dialog executed.")
        except Exception as e:
            logging.error(f"Failed to print page: {e}")

    def scrape_page(self):
        try:
            self.browser.page().toHtml(self.handle_scrape_result)
            logging.info("Scrape page initiated.")
        except Exception as e:
            logging.error(f"Failed to scrape page: {e}")

    def handle_scrape_result(self, html):
        try:
            logging.info("Scraped page content")
            print(html)
        except Exception as e:
            logging.error(f"Failed to handle scrape result: {e}")

    def closeEvent(self, event):
        try:
            event.accept()
            logging.info("Application closed.")
        except Exception as e:
            logging.error(f"Failed to close application: {e}")


class WebPageHandler:
    def __init__(self, browser):
        self.browser = browser
        self.connect_signals()

    def connect_signals(self):
        self.browser.urlChanged.connect(self.update_url)
        self.browser.loadFinished.connect(self.update_title)
        self.browser.loadFinished.connect(self.on_load_finished)
        self.browser.loadProgress.connect(self.on_load_progress)
        self.browser.loadStarted.connect(self.on_load_started)

    def on_load_started(self):
        self.browser.parent().status.showMessage("Loading started...")
        logging.info("Loading started...")

    def on_load_progress(self, progress):
        self.browser.parent().status.showMessage(f"Loading... {progress}%")
        logging.info(f"Loading progress: {progress}%")

    def on_load_finished(self, ok):
        if not ok:
            logging.error("Failed to load the page.")
            self.browser.parent().status.showMessage("Failed to load the page.")
        else:
            self.browser.parent().status.showMessage("Loading finished.")
            logging.info("Loading finished.")

    def update_url(self, q):
        try:
            self.browser.parent().url_bar.setText(q.toString())
            logging.info(f"URL updated to: {q.toString()}")
        except Exception as e:
            logging.error(f"Failed to update URL: {e}")

    def update_title(self):
        try:
            title = self.browser.page().title()
            self.browser.parent().setWindowTitle(f"{title} - Advanced Web Browser")
            logging.info(f"Title updated to: {title}")
        except Exception as e:
            logging.error(f"Failed to update title: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrowserUI()
    window.browser = QWebEngineView()
    window.browser.setPage(CustomWebEnginePage(window.browser))
    window.browser.setUrl(QUrl("http://www.google.com"))
    window.layout.addWidget(window.browser)
    WebPageHandler(window.browser)
    window.show()
    sys.exit(app.exec_())
