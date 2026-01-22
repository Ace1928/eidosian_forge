import matplotlib.pyplot as plt
import numpy as np
def _cumulative_gain_curve(y_true, y_score, pos_label=None):
    """
    This method is copied from scikit-plot package.
    See https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/helpers.py#L157

    This function generates the points necessary to plot the Cumulative Gain

    Note: This implementation is restricted to the binary classification task.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).

        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative

    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.

        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = (np.asarray(y_true), np.asarray(y_score))
    classes = np.unique(y_true)
    if pos_label is None and (not (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1]) or np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or np.array_equal(classes, [1]))):
        raise ValueError('Data is not binary and pos_label is not specified')
    elif pos_label is None:
        pos_label = 1.0
    y_true = y_true == pos_label
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)
    percentages = np.arange(start=1, stop=len(y_true) + 1)
    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))
    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])
    return (percentages, gains)