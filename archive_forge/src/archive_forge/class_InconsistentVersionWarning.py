class InconsistentVersionWarning(UserWarning):
    """Warning raised when an estimator is unpickled with a inconsistent version.

    Parameters
    ----------
    estimator_name : str
        Estimator name.

    current_sklearn_version : str
        Current scikit-learn version.

    original_sklearn_version : str
        Original scikit-learn version.
    """

    def __init__(self, *, estimator_name, current_sklearn_version, original_sklearn_version):
        self.estimator_name = estimator_name
        self.current_sklearn_version = current_sklearn_version
        self.original_sklearn_version = original_sklearn_version

    def __str__(self):
        return f'Trying to unpickle estimator {self.estimator_name} from version {self.original_sklearn_version} when using version {self.current_sklearn_version}. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\nhttps://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations'