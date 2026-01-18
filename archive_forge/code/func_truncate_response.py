import asyncio
import os
import threading
from threading import Event
from typing import Optional
import discord
import gradio as gr
from discord import Permissions
from discord.ext import commands
from discord.utils import oauth_url
import gradio_client as grc
from gradio_client.utils import QueueError
def truncate_response(response: str) -> str:
    ending = '...\nTruncating response to 2000 characters due to discord api limits.'
    if len(response) > 2000:
        return response[:2000 - len(ending)] + ending
    else:
        return response