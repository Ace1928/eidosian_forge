from abc import ABC, abstractmethod
from fugue.dataframe import DataFrame
from fugue.extensions.context import ExtensionContext
Create DataFrame on driver side

        .. note::

          * It runs on driver side
          * The output dataframe is not necessarily local, for example a SparkDataFrame
          * It is engine aware, you can put platform dependent code in it (for example
            native pyspark code) but by doing so your code may not be portable. If you
            only use the functions of the general ExecutionEngine interface, it's still
            portable.

        :return: result dataframe
        