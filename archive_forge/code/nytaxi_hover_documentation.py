import numpy as np
import holoviews as hv
import dask.dataframe as dd
from holoviews import opts
from holoviews.operation.datashader import aggregate

Bokeh app example using datashader for rasterizing a large dataset and
geoviews for reprojecting coordinate systems.

This example requires the 1.7GB nyc_taxi_wide.parquet dataset which
you can obtain by downloading the file from AWS:

  https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq

Place this parquet in a data/ subfolder and install the python dependencies, e.g.

  conda install datashader fastparquet python-snappy

You can now run this app with:

  bokeh serve --show nytaxi_hover.py

