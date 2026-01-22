from __future__ import annotations
import hashlib
import os
import typing
import urllib.parse
import warnings
from dataclasses import dataclass, field
import fastapi
from fastapi.responses import RedirectResponse
from huggingface_hub import HfFolder, whoami
from .utils import get_space
def _add_mocked_oauth_routes(app: fastapi.FastAPI) -> None:
    """Add fake oauth routes if Gradio is run locally and OAuth is enabled.

    Clicking on a gr.LoginButton will have the same behavior as in a Space (i.e. gets redirected in a new tab) but
    instead of authenticating with HF, a mocked user profile is added to the session.
    """
    warnings.warn('Gradio does not support OAuth features outside of a Space environment. To help you debug your app locally, the login and logout buttons are mocked with your profile. To make it work, your machine must be logged in to Huggingface.')
    mocked_oauth_info = _get_mocked_oauth_info()

    @app.get('/login/huggingface')
    async def oauth_login(request: fastapi.Request):
        """Fake endpoint that redirects to HF OAuth page."""
        redirect_uri = _generate_redirect_uri(request)
        return RedirectResponse('/login/callback?' + urllib.parse.urlencode({'_target_url': redirect_uri}))

    @app.get('/login/callback')
    async def oauth_redirect_callback(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that handles the OAuth callback."""
        request.session['oauth_info'] = mocked_oauth_info
        return _redirect_to_target(request)

    @app.get('/logout')
    async def oauth_logout(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that logs out the user (e.g. delete cookie session)."""
        request.session.pop('oauth_info', None)
        logout_url = str(request.url).replace('/logout', '/')
        return RedirectResponse(url=logout_url)