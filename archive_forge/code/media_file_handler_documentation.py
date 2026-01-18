from __future__ import annotations
from urllib.parse import quote
import tornado.web
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorageError
from streamlit.runtime.memory_media_file_storage import (
from streamlit.web.server import allow_cross_origin_requests
Add Content-Disposition header for downloadable files.

        Set header value to "attachment" indicating that file should be saved
        locally instead of displaying inline in browser.

        We also set filename to specify the filename for downloaded files.
        Used for serving downloadable files, like files stored via the
        `st.download_button` widget.
        