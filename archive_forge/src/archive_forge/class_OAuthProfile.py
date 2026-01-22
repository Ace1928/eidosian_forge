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
@dataclass
class OAuthProfile(typing.Dict):
    """
    A Gradio OAuthProfile object that can be used to inject the profile of a user in a
    function. If a function expects `OAuthProfile` or `Optional[OAuthProfile]` as input,
    the value will be injected from the FastAPI session if the user is logged in. If the
    user is not logged in and the function expects `OAuthProfile`, an error will be
    raised.

    Attributes:
        name (str): The name of the user (e.g. 'Abubakar Abid').
        username (str): The username of the user (e.g. 'abidlabs')
        profile (str): The profile URL of the user (e.g. 'https://huggingface.co/abidlabs').
        picture (str): The profile picture URL of the user.

    Example:
        import gradio as gr
        from typing import Optional


        def hello(profile: Optional[gr.OAuthProfile]) -> str:
            if profile is None:
                return "I don't know you."
            return f"Hello {profile.name}"


        with gr.Blocks() as demo:
            gr.LoginButton()
            gr.Markdown().attach_load_event(hello, None)
    """
    name: str = field(init=False)
    username: str = field(init=False)
    profile: str = field(init=False)
    picture: str = field(init=False)

    def __init__(self, data: dict):
        self.update(data)
        self.name = self['name']
        self.username = self['preferred_username']
        self.profile = self['profile']
        self.picture = self['picture']