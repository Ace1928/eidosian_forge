import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def upload_to_hub(data):
    try:
        theme_url = data[current_theme].push_to_hub(repo_name=data[theme_name], version=data[theme_version] or None, hf_token=data[theme_hf_token], theme_name=data[theme_name])
        space_name = '/'.join(theme_url.split('/')[-2:])
        return (gr.Markdown(value=f"Theme uploaded [here!]({theme_url})! Load it as `gr.Blocks(theme='{space_name}')`", visible=True), 'Upload to Hub')
    except Exception as e:
        return (gr.Markdown(value=f'Error: {e}', visible=True), 'Upload to Hub')