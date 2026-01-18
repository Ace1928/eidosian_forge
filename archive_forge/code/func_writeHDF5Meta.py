import copy
import os
import pickle
import warnings
import numpy as np
def writeHDF5Meta(self, root, name, data, **dsOpts):
    if isinstance(data, np.ndarray):
        dsOpts['maxshape'] = (None,) + data.shape[1:]
        root.create_dataset(name, data=data, **dsOpts)
    elif isinstance(data, list) or isinstance(data, tuple):
        gr = root.create_group(name)
        if isinstance(data, list):
            gr.attrs['_metaType_'] = 'list'
        else:
            gr.attrs['_metaType_'] = 'tuple'
        for i in range(len(data)):
            self.writeHDF5Meta(gr, str(i), data[i], **dsOpts)
    elif isinstance(data, dict):
        gr = root.create_group(name)
        gr.attrs['_metaType_'] = 'dict'
        for k, v in data.items():
            self.writeHDF5Meta(gr, k, v, **dsOpts)
    elif isinstance(data, int) or isinstance(data, float) or isinstance(data, np.integer) or isinstance(data, np.floating):
        root.attrs[name] = data
    else:
        try:
            root.attrs[name] = repr(data)
        except:
            print("Can not store meta data of type '%s' in HDF5. (key is '%s')" % (str(type(data)), str(name)))
            raise