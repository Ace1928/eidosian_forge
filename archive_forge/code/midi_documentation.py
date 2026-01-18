import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth

    Install fluidsynth first, see more: https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
    