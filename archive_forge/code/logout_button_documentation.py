from __future__ import annotations
import warnings
from typing import Literal
from gradio_client.documentation import document
from gradio.components import Button

    Creates a Button to log out a user from a Space using OAuth.

    Note: `LogoutButton` component is deprecated. Please use `gr.LoginButton` instead
          which handles both the login and logout processes.
    