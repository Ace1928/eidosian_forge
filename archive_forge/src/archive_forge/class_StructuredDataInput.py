from typing import Dict
from typing import List
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import adapters
from autokeras import analysers
from autokeras import blocks
from autokeras import hyper_preprocessors as hpps_module
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module
class StructuredDataInput(Input):
    """Input node for structured data.

    The input data should be numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
    The data should be two-dimensional with numerical or categorical values.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data. A column will be judged as
            categorical if the number of different values is less than 5% of the
            number of instances.
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, column_names: Optional[List[str]]=None, column_types: Optional[Dict[str, str]]=None, name: Optional[str]=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.column_names = column_names
        self.column_types = column_types

    def get_config(self):
        config = super().get_config()
        config.update({'column_names': self.column_names, 'column_types': self.column_types})
        return config

    def get_adapter(self):
        return adapters.StructuredDataAdapter()

    def get_analyser(self):
        return analysers.StructuredDataAnalyser(self.column_names, self.column_types)

    def get_block(self):
        return blocks.StructuredDataBlock()

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.column_names = analyser.column_names
        self.column_types = analyser.column_types

    def build(self, hp, inputs=None):
        return inputs