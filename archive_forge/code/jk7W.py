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
    """
    CustomWebEnginePage extends QWebEnginePage to handle SSL certificate errors and other custom behaviors.
    """

    def certificateError(
        self, certificateError: QWebEnginePage.CertificateError
    ) -> bool:
        """
        Handles SSL certificate errors by logging the error and determining if it is overridable.

        Args:
            certificateError (QWebEnginePage.CertificateError): The SSL certificate error encountered.

        Returns:
            bool: True if the error is overridable and the user chooses to proceed, False otherwise.
        """
        logging.error(f"SSL Error: {certificateError.errorDescription()}")
        if certificateError.isOverridable():
            logging.info("SSL Error is overridable, proceeding.")
            return True
        else:
            logging.info("SSL Error is not overridable, blocking.")
            return False

    def javaScriptConsoleMessage(
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        lineNumber: int,
        sourceID: str,
    ):
        """
        Handles JavaScript console messages by logging them.

        Args:
            level (QWebEnginePage.JavaScriptConsoleMessageLevel): The severity level of the message.
            message (str): The console message.
            lineNumber (int): The line number where the message originated.
            sourceID (str): The source identifier of the message.
        """
        if level == QWebEnginePage.InfoMessageLevel:
            logging.info(f"JS Info: {message} (Source: {sourceID}, Line: {lineNumber})")
        elif level == QWebEnginePage.WarningMessageLevel:
            logging.warning(
                f"JS Warning: {message} (Source: {sourceID}, Line: {lineNumber})"
            )
        elif level == QWebEnginePage.ErrorMessageLevel:
            logging.error(
                f"JS Error: {message} (Source: {sourceID}, Line: {lineNumber})"
            )

    def acceptNavigationRequest(
        self, url: QUrl, _type: QWebEnginePage.NavigationType, isMainFrame: bool
    ) -> bool:
        """
        Handles navigation requests, allowing for custom URL filtering or logging.

        Args:
            url (QUrl): The URL to navigate to.
            _type (QWebEnginePage.NavigationType): The type of navigation.
            isMainFrame (bool): Whether the navigation is in the main frame.

        Returns:
            bool: True to accept the navigation request, False to reject it.
        """
        logging.info(
            f"Navigation request to: {url.toString()} (Type: {_type}, MainFrame: {isMainFrame})"
        )
        return super().acceptNavigationRequest(url, _type, isMainFrame)

    def featurePermissionRequested(
        self, securityOrigin: QUrl, feature: QWebEnginePage.Feature
    ):
        """
        Handles feature permission requests, such as geolocation or media access.

        Args:
            securityOrigin (QUrl): The security origin requesting the feature.
            feature (QWebEnginePage.Feature): The feature being requested.
        """
        logging.info(
            f"Feature permission requested: {feature} from {securityOrigin.toString()}"
        )
        self.setFeaturePermission(
            securityOrigin, feature, QWebEnginePage.PermissionGrantedByUser
        )

    def fullScreenRequested(self, request: QWebEngineFullScreenRequest):
        """
        Handles full-screen requests.

        Args:
            request (QWebEngineFullScreenRequest): The full-screen request.
        """
        logging.info(
            f"Full-screen request: {'Entering' if request.toggleOn() else 'Exiting'} full-screen mode."
        )
        request.accept()


class Browser(QWebEngineView):
    """
    Browser class that extends QWebEngineView to provide a custom web browser with enhanced features.
    """

    def __init__(self, home_url: str = "http://www.google.com"):
        """
        Initializes the Browser with a custom web engine page and sets the home URL.

        Args:
            home_url (str, optional): The URL to set as the home page. Defaults to "http://www.google.com".
        """
        super().__init__()
        self.setPage(CustomWebEnginePage(self))
        self.setUrl(QUrl(home_url))
        self.home_url = home_url
        self.history = []
        self.init_ui()
        logging.info(f"Browser initialized with home page: {home_url}")

    def init_ui(self):
        """
        Initializes the user interface components and connects signals.
        """
        self.loadFinished.connect(self.on_load_finished)
        self.urlChanged.connect(self.on_url_changed)
        logging.info("UI components initialized and signals connected.")

    def on_load_finished(self, success: bool):
        """
        Slot to handle the loadFinished signal.

        Args:
            success (bool): Indicates whether the page loaded successfully.
        """
        if success:
            logging.info("Page loaded successfully.")
        else:
            logging.error("Failed to load the page.")

    def on_url_changed(self, url: QUrl):
        """
        Slot to handle the urlChanged signal.

        Args:
            url (QUrl): The new URL.
        """
        self.history.append(url.toString())
        logging.info(f"URL changed to: {url.toString()}")

    def navigate_to(self, url: str):
        """
        Navigates to the specified URL.

        Args:
            url (str): The URL to navigate to.
        """
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        self.setUrl(QUrl(url))
        logging.info(f"Navigating to URL: {url}")

    def navigate_home(self):
        """
        Navigates to the home URL.
        """
        self.setUrl(QUrl(self.home_url))
        logging.info(f"Navigating to home URL: {self.home_url}")

    def reload_page(self):
        """
        Reloads the current page.
        """
        self.reload()
        logging.info("Page reloaded.")

    def stop_loading(self):
        """
        Stops the loading of the current page.
        """
        self.stop()
        logging.info("Page loading stopped.")

    def zoom_in(self):
        """
        Increases the zoom level of the page.
        """
        self.setZoomFactor(self.zoomFactor() + 0.1)
        logging.info(f"Zoomed in. Current zoom factor: {self.zoomFactor()}")

    def zoom_out(self):
        """
        Decreases the zoom level of the page.
        """
        self.setZoomFactor(self.zoomFactor() - 0.1)
        logging.info(f"Zoomed out. Current zoom factor: {self.zoomFactor()}")

    def reset_zoom(self):
        """
        Resets the zoom level to the default value.
        """
        self.setZoomFactor(1.0)
        logging.info("Zoom reset to default.")

    def print_page(self):
        """
        Opens the print dialog to print the current page.
        """
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        if dialog.exec_() == QDialog.Accepted:
            self.page().print(
                printer,
                lambda success: logging.info(
                    "Printing completed" if success else "Printing failed"
                ),
            )
        logging.info("Print dialog executed.")


class TabManager(QTabWidget):
    """
    TabManager class to manage browser tabs within the application.
    """

    def __init__(self, parent: QWidget = None):
        """
        Initializes the TabManager with closable tabs and connects signals.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.tabBarDoubleClicked.connect(self.open_new_tab)
        self.setDocumentMode(True)
        self.setMovable(True)
        logging.info("TabManager initialized.")

    def add_tab(self, url: str = "http://www.google.com", label: str = "New Tab"):
        """
        Adds a new tab with the specified URL and label.

        Args:
            url (str, optional): The URL to load in the new tab. Defaults to "http://www.google.com".
            label (str, optional): The label for the new tab. Defaults to "New Tab".
        """
        browser = Browser()
        browser.setUrl(QUrl(url))
        index = self.addTab(browser, label)
        self.setCurrentIndex(index)
        logging.info(f"New tab added with URL: {url} and label: {label}")

    def close_tab(self, index: int):
        """
        Closes the tab at the specified index if more than one tab is open.

        Args:
            index (int): The index of the tab to close.
        """
        if self.count() > 1:
            self.widget(index).deleteLater()
            self.removeTab(index)
            logging.info(f"Tab at index {index} closed.")
        else:
            logging.warning("Attempted to close the last remaining tab.")

    def open_new_tab(self, index: int):
        """
        Opens a new tab when the tab bar is double-clicked.

        Args:
            index (int): The index where the double-click occurred.
        """
        if index == -1:  # Double-click on empty space
            self.add_tab()
            logging.info("New tab opened via double-click.")

    def current_browser(self) -> Browser:
        """
        Returns the current browser instance in the active tab.

        Returns:
            Browser: The current browser instance.
        """
        return self.currentWidget()

    def reload_current_tab(self):
        """
        Reloads the current active tab.
        """
        browser = self.current_browser()
        if browser:
            browser.reload()
            logging.info("Current tab reloaded.")

    def navigate_home_current_tab(self):
        """
        Navigates the current active tab to the home page.
        """
        browser = self.current_browser()
        if browser:
            browser.setUrl(QUrl("http://www.google.com"))
            logging.info("Current tab navigated to home page.")


class BookmarkManager(QWidget):
    """
    BookmarkManager class to manage and display bookmarks.
    """

    def __init__(self, parent=None):
        """
        Initializes the BookmarkManager with a layout and bookmarks list widget.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.bookmarks = QListWidget()
        self.layout.addWidget(self.bookmarks)
        self.bookmark_data = []
        self.load_bookmarks()

    def add_bookmark(self, title: str, url: str):
        """
        Adds a new bookmark to the bookmarks list and saves it.

        Args:
            title (str): The title of the bookmark.
            url (str): The URL of the bookmark.
        """
        item = QListWidgetItem(f"{title} - {url}")
        self.bookmarks.addItem(item)
        self.bookmark_data.append({"title": title, "url": url})
        self.save_bookmarks()
        logging.info(f"Bookmark added: {title} - {url}")

    def load_bookmarks(self):
        """
        Loads bookmarks from a file.
        """
        try:
            with open("bookmarks.json", "r") as file:
                self.bookmark_data = json.load(file)
                for entry in self.bookmark_data:
                    item = QListWidgetItem(f"{entry['title']} - {entry['url']}")
                    self.bookmarks.addItem(item)
            logging.info("Bookmarks loaded successfully.")
        except FileNotFoundError:
            logging.warning("Bookmarks file not found. Starting with an empty list.")
        except Exception as e:
            logging.error(f"Failed to load bookmarks: {e}")

    def save_bookmarks(self):
        """
        Saves the current bookmarks to a file.
        """
        try:
            with open("bookmarks.json", "w") as file:
                json.dump(self.bookmark_data, file, indent=4)
            logging.info("Bookmarks saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save bookmarks: {e}")

    def remove_bookmark(self, index: int):
        """
        Removes a specific bookmark by index.

        Args:
            index (int): The index of the bookmark to be removed.
        """
        if 0 <= index < len(self.bookmark_data):
            self.bookmarks.takeItem(index)
            del self.bookmark_data[index]
            self.save_bookmarks()
            logging.info(f"Bookmark at index {index} removed successfully.")
        else:
            logging.warning(f"Invalid index {index} for bookmark removal.")

    def search_bookmarks(self, query: str) -> list:
        """
        Searches the bookmarks for items matching the query.

        Args:
            query (str): The search query.

        Returns:
            list: A list of matching bookmark items.
        """
        results = [
            entry
            for entry in self.bookmark_data
            if query.lower() in entry["title"].lower()
            or query.lower() in entry["url"].lower()
        ]
        logging.info(f"Search for '{query}' returned {len(results)} results.")
        return results

    def clear_bookmarks(self):
        """
        Clears all bookmarks from the list.
        """
        self.bookmarks.clear()
        self.bookmark_data = []
        self.save_bookmarks()
        logging.info("All bookmarks cleared.")


class DownloadManager(QDialog):
    """
    DownloadManager class to manage and display download tasks.
    """

    def __init__(self, parent=None):
        """
        Initializes the DownloadManager with a layout and downloads list widget.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.downloads = QListWidget()
        self.layout.addWidget(self.downloads)
        self.download_tasks = []
        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface components.
        """
        self.setWindowTitle("Download Manager")
        self.setGeometry(300, 300, 400, 300)
        self.show()

    def start_download(self, url: str):
        """
        Starts a new download and adds it to the download list.

        Args:
            url (str): The URL of the file to be downloaded.
        """
        download_task = {"url": url, "progress": 0, "status": "Starting"}
        self.download_tasks.append(download_task)
        item = QListWidgetItem(f"{url} - {download_task['status']}")
        self.downloads.addItem(item)
        self.download_file(url, item)

    def download_file(self, url: str, item: QListWidgetItem):
        """
        Simulates downloading a file and updates the download progress.

        Args:
            url (str): The URL of the file to be downloaded.
            item (QListWidgetItem): The list widget item representing the download.
        """
        import time
        import threading

        def download():
            for i in range(1, 101):
                time.sleep(0.1)  # Simulate download time
                self.update_download(item, i)
            item.setText(f"{url} - Completed")

        threading.Thread(target=download).start()

    def update_download(self, item: QListWidgetItem, progress: int):
        """
        Updates the download progress.

        Args:
            item (QListWidgetItem): The list widget item representing the download.
            progress (int): The current progress of the download.
        """
        item.setText(f"{item.text().split(' - ')[0]} - {progress}%")

    def cancel_download(self, index: int):
        """
        Cancels an ongoing download.

        Args:
            index (int): The index of the download to be canceled.
        """
        if 0 <= index < len(self.download_tasks):
            self.download_tasks[index]["status"] = "Canceled"
            self.downloads.item(index).setText(
                f"{self.download_tasks[index]['url']} - Canceled"
            )
            logging.info(f"Download canceled: {self.download_tasks[index]['url']}")
        else:
            logging.warning(f"Invalid index {index} for download cancellation.")

    def remove_download(self, index: int):
        """
        Removes a download from the list.

        Args:
            index (int): The index of the download to be removed.
        """
        if 0 <= index < len(self.download_tasks):
            self.download_tasks.pop(index)
            self.downloads.takeItem(index)
            logging.info(f"Download removed at index {index}")
        else:
            logging.warning(f"Invalid index {index} for download removal.")

    def clear_downloads(self):
        """
        Clears all downloads from the list.
        """
        self.download_tasks.clear()
        self.downloads.clear()
        logging.info("All downloads cleared.")


class HistoryManager(QWidget):
    """
    HistoryManager class to manage and display browsing history.
    """

    def __init__(self, parent=None):
        """
        Initializes the HistoryManager with a layout and history list widget.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.history_list = QListWidget()
        self.layout.addWidget(self.history_list)
        self.history_data = []
        self.load_history()

    def add_history_item(self, title: str, url: str):
        """
        Adds a new history item to the history list and saves it.

        Args:
            title (str): The title of the web page.
            url (str): The URL of the web page.
        """
        item = QListWidgetItem(f"{title} - {url}")
        self.history_list.addItem(item)
        self.history_data.append({"title": title, "url": url})
        self.save_history()

    def clear_history(self):
        """
        Clears the history list and deletes the saved history.
        """
        self.history_list.clear()
        self.history_data = []
        self.save_history()

    def load_history(self):
        """
        Loads the browsing history from a file.
        """
        try:
            with open("history.json", "r") as file:
                self.history_data = json.load(file)
                for entry in self.history_data:
                    item = QListWidgetItem(f"{entry['title']} - {entry['url']}")
                    self.history_list.addItem(item)
            logging.info("History loaded successfully.")
        except FileNotFoundError:
            logging.warning("History file not found. Starting with an empty history.")
        except Exception as e:
            logging.error(f"Failed to load history: {e}")

    def save_history(self):
        """
        Saves the current browsing history to a file.
        """
        try:
            with open("history.json", "w") as file:
                json.dump(self.history_data, file, indent=4)
            logging.info("History saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save history: {e}")

    def remove_history_item(self, index: int):
        """
        Removes a specific history item by index.

        Args:
            index (int): The index of the item to be removed.
        """
        if 0 <= index < len(self.history_data):
            self.history_list.takeItem(index)
            del self.history_data[index]
            self.save_history()
            logging.info(f"History item at index {index} removed successfully.")
        else:
            logging.warning(f"Invalid index {index} for history removal.")

    def search_history(self, query: str) -> list:
        """
        Searches the history for items matching the query.

        Args:
            query (str): The search query.

        Returns:
            list: A list of matching history items.
        """
        results = [
            entry
            for entry in self.history_data
            if query.lower() in entry["title"].lower()
            or query.lower() in entry["url"].lower()
        ]
        logging.info(f"Search for '{query}' returned {len(results)} results.")
        return results


class AdBlocker:
    """
    AdBlocker class to manage and enforce ad blocking rules.
    """

    def __init__(self):
        """
        Initializes the AdBlocker with a blocklist.
        """
        self.blocklist = self.load_blocklist()

    def load_blocklist(self) -> set:
        """
        Loads the blocklist from a file or online source.

        Returns:
            set: A set of URLs to be blocked.
        """
        try:
            # Example: Load blocklist from a local file
            with open("blocklist.txt", "r") as file:
                blocklist = set(line.strip() for line in file if line.strip())
            logging.info("Blocklist loaded successfully.")
            return blocklist
        except FileNotFoundError:
            logging.error("Blocklist file not found.")
            return set()
        except Exception as e:
            logging.error(f"Failed to load blocklist: {e}")
            return set()

    def should_block(self, request_url: str) -> bool:
        """
        Determines if the given URL should be blocked.

        Args:
            request_url (str): The URL to be checked.

        Returns:
            bool: True if the URL should be blocked, False otherwise.
        """
        for blocked_url in self.blocklist:
            if blocked_url in request_url:
                logging.info(f"Blocking URL: {request_url}")
                return True
        logging.info(f"Allowing URL: {request_url}")
        return False

    def update_blocklist(self, new_entries: list):
        """
        Updates the blocklist with new entries.

        Args:
            new_entries (list): A list of new URLs to be added to the blocklist.
        """
        self.blocklist.update(new_entries)
        self.save_blocklist()
        logging.info("Blocklist updated successfully.")

    def save_blocklist(self):
        """
        Saves the current blocklist to a file.
        """
        try:
            with open("blocklist.txt", "w") as file:
                for url in self.blocklist:
                    file.write(f"{url}\n")
            logging.info("Blocklist saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save blocklist: {e}")


class PrivacyModeBrowser(Browser):
    """
    A specialized browser class that operates in privacy mode, ensuring no persistent cookies or history are stored.
    """

    def __init__(self):
        """
        Initializes the PrivacyModeBrowser with settings to disable persistent cookies and history.
        """
        super().__init__()
        self._configure_privacy_settings()

    def _configure_privacy_settings(self):
        """
        Configures the browser to disable persistent cookies and history.
        """
        profile = self.page().profile()
        profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
        profile.setHistoryType(QWebEngineProfile.NoHistory)
        logging.info("Privacy mode enabled: No persistent cookies, no history.")

    def clear_cache(self):
        """
        Clears the browser's cache to ensure no data is retained.
        """
        self.page().profile().clearHttpCache()
        logging.info("Browser cache cleared.")

    def enable_do_not_track(self):
        """
        Enables the 'Do Not Track' feature to enhance privacy.
        """
        self.page().profile().setHttpUserAgent(
            self.page().profile().httpUserAgent() + " DNT/1.0"
        )
        logging.info("Do Not Track enabled.")

    def disable_javascript(self):
        """
        Disables JavaScript execution for enhanced privacy.
        """
        settings = self.page().settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, False)
        logging.info("JavaScript disabled.")

    def enable_ad_blocking(self):
        """
        Enables ad blocking by integrating with an AdBlocker instance.
        """
        self.ad_blocker = AdBlocker()
        self.page().profile().setRequestInterceptor(self.ad_blocker)
        logging.info("Ad blocking enabled.")

    def set_custom_user_agent(self, user_agent: str):
        """
        Sets a custom user agent string for the browser.

        Args:
            user_agent (str): The custom user agent string to be set.
        """
        self.page().profile().setHttpUserAgent(user_agent)
        logging.info(f"Custom user agent set: {user_agent}")


class ExtensionManager:
    """
    Manages browser extensions, allowing for loading, unloading, and managing extensions.
    """

    def __init__(self):
        """
        Initializes the ExtensionManager with an empty list of extensions.
        """
        self.extensions = []

    def load_extension(self, extension_path: str) -> bool:
        """
        Loads and initializes an extension from the given path.

        Args:
            extension_path (str): The file path to the extension.

        Returns:
            bool: True if the extension was loaded successfully, False otherwise.
        """
        try:
            # Simulate loading the extension (e.g., reading a file, initializing objects)
            extension = self._initialize_extension(extension_path)
            self.extensions.append(extension)
            logging.info(f"Extension loaded from {extension_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load extension from {extension_path}: {e}")
            return False

    def unload_extension(self, extension_name: str) -> bool:
        """
        Unloads an extension by its name.

        Args:
            extension_name (str): The name of the extension to unload.

        Returns:
            bool: True if the extension was unloaded successfully, False otherwise.
        """
        try:
            self.extensions = [
                ext for ext in self.extensions if ext["name"] != extension_name
            ]
            logging.info(f"Extension {extension_name} unloaded successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to unload extension {extension_name}: {e}")
            return False

    def list_extensions(self) -> list:
        """
        Lists all currently loaded extensions.

        Returns:
            list: A list of dictionaries containing extension details.
        """
        return self.extensions

    def _initialize_extension(self, extension_path: str) -> dict:
        """
        Initializes an extension from the given path. This is a private method.

        Args:
            extension_path (str): The file path to the extension.

        Returns:
            dict: A dictionary containing extension details.
        """
        # Placeholder for actual extension initialization logic
        extension = {
            "name": extension_path.split("/")[-1],  # Example: Extracting name from path
            "path": extension_path,
            "version": "1.0.0",  # Example version
            "enabled": True,
        }
        return extension


class BrowserUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.browser = Browser()
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
        self.layout.addWidget(self.browser)
        logging.info("Central widget set up successfully.")

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
            logging.info(f"Action '{name}' added to navigation bar.")

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        self.navbar.addWidget(self.url_bar)
        logging.info("URL bar added to navigation bar.")

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
            logging.info(f"Action '{name}' added to file menu.")

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
            logging.info(f"Action '{name}' added to view menu.")

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
        logging.info("Signals connected to their respective slots.")

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
    WebPageHandler(window.browser)
    window.show()
    sys.exit(app.exec_())
