import json
import os
import re
import sys
import numpy as np
def CreateDictFromFlatbuffer(buffer_data):
    model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    return FlatbufferToDict(model, preserve_as_numpy=False)