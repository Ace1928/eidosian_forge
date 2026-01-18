import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def volume_mapper_builder(state, zf, register):
    instance = generic_builder(state, zf, register)
    instance.SetScalarMode(1)