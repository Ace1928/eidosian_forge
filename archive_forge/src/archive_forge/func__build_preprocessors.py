import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import preprocessors as preprocessors_module
from autokeras.engine import hyper_preprocessor as hpps_module
from autokeras.engine import preprocessor as pps_module
from autokeras.utils import data_utils
from autokeras.utils import io_utils
@staticmethod
def _build_preprocessors(hp, hpps_lists, dataset):
    sources = data_utils.unzip_dataset(dataset)
    preprocessors_list = []
    for source, hpps_list in zip(sources, hpps_lists):
        data = source
        preprocessors = []
        for hyper_preprocessor in hpps_list:
            preprocessor = hyper_preprocessor.build(hp, data)
            preprocessor.fit(data)
            data = preprocessor.transform(data)
            preprocessors.append(preprocessor)
        preprocessors_list.append(preprocessors)
    return preprocessors_list