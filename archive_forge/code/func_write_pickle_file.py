import os
import pickle
from traits.api import Float
from traits import __version__
def write_pickle_file(description, picklee, protocol, output_dir):
    filename = PKL_FILENAME.format(traits_version=__version__, pickle_protocol=protocol, description=description)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as pickle_file:
        pickle.dump(picklee, pickle_file, protocol=protocol)