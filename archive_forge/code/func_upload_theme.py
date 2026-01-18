from __future__ import annotations
import argparse
from gradio.themes import ThemeClass
def upload_theme(theme: str, repo_name: str, org_name: str | None=None, version: str | None=None, hf_token: str | None=None, theme_name: str | None=None, description: str | None=None):
    theme = ThemeClass.load(theme)
    return theme.push_to_hub(repo_name=repo_name, version=version, hf_token=hf_token, theme_name=theme_name, description=description, org_name=org_name)