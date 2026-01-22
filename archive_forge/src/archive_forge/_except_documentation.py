from typing import Optional

        @type reason: L{bytes}
        @param reason: The server response minus the status indicator.

        @type consumer: callable that takes L{object}
        @param consumer: The function meant to handle the values for a
            multi-line response.
        