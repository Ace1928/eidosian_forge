import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth
@router.post('/txt-to-midi', tags=['MIDI'])
def txt_to_midi(body: TxtToMidiBody):
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    if not body.midi_path.startswith('midi/'):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'bad output path')
    vocab_config_type = global_var.get(global_var.Midi_Vocab_Config_Type)
    if vocab_config_type == global_var.MidiVocabConfig.Piano:
        vocab_config = 'backend-python/utils/vocab_config_piano.json'
    else:
        vocab_config = 'backend-python/utils/midi_vocab_config.json'
    cfg = VocabConfig.from_json(vocab_config)
    with open(body.txt_path, 'r') as f:
        text = f.read()
    text = text.strip()
    mid = convert_str_to_midi(cfg, text)
    mid.save(body.midi_path)
    return 'success'