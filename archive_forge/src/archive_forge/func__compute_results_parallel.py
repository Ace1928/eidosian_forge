import time
from warnings import simplefilter
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
import wandb
def _compute_results_parallel(n_jobs, clusterer, X, cluster_ranges):
    parallel_runner = Parallel(n_jobs=n_jobs)
    _cluster_scorer = delayed(_clone_and_score_clusterer)
    results = parallel_runner((_cluster_scorer(clusterer, X, i) for i in cluster_ranges))
    clfs, times = zip(*results)
    return (clfs, times)