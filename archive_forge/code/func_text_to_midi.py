import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth
@router.post('/text-to-midi', tags=['MIDI'])
def text_to_midi(body: TextToMidiBody):
    vocab_config_type = global_var.get(global_var.Midi_Vocab_Config_Type)
    if vocab_config_type == global_var.MidiVocabConfig.Piano:
        vocab_config = 'backend-python/utils/vocab_config_piano.json'
    else:
        vocab_config = 'backend-python/utils/midi_vocab_config.json'
    cfg = VocabConfig.from_json(vocab_config)
    mid = convert_str_to_midi(cfg, body.text.strip())
    mid_data = io.BytesIO()
    mid.save(None, mid_data)
    mid_data.seek(0)
    return StreamingResponse(mid_data, media_type='audio/midi')