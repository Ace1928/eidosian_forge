def _get_dataset(d):
    import pandas
    import os
    return pandas.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'package_data', 'datasets', d + '.csv.gz'))