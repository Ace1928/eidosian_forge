################################################################################
# Standard library imports
################################################################################
import os
import sys
import math
import time
import json
import heapq
import logging
import platform
import traceback
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union, Any, Iterable, Callable

################################################################################
# Third-party imports
################################################################################
import numpy as np
from numpy.random import uniform
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Scientific computing
from scipy import stats
from scipy.stats import entropy, norm, t
from scipy.optimize import curve_fit

# Machine learning
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# Performance optimization
import numba
from numba import jit, prange, objmode

################################################################################
# Logging Configuration
################################################################################
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eidos_sort.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

################################################################################
# Global Output Directory and Parameters
################################################################################
output_dir = Path("benchmark_results")
output_dir.mkdir(parents=True, exist_ok=True)

params_file = output_dir / "eidos_params.json"
if params_file.exists():
    with open(params_file) as f:
        ADAPTIVE_PARAMS = json.load(f)
else:
    ADAPTIVE_PARAMS = {
        "entropy_threshold": 6.0,
        "cluster_size_threshold": 5000,
        "bit_analysis_threshold": 500,
        "chunk_size": 1000,
        "insertion_threshold": 16,
        "success_rate": 0.0,
        "total_runs": 0
    }

################################################################################
# Progress Tracking
################################################################################
def track_progress(
    iterable: Iterable,
    desc: Optional[str] = None,
    total: Optional[int] = None
) -> Iterable:
    """
    Wrapper for the tqdm progress bar with enhanced formatting.

    Args:
        iterable (Iterable): The data or range over which to iterate.
        desc (Optional[str]): Optional description for the progress bar.
        total (Optional[int]): Optional total length for the progress bar.

    Returns:
        Iterable: The same iterable but wrapped with tqdm progress tracking.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

################################################################################
# Enhanced Analysis Functions
################################################################################
def analyze_data_patterns(data: np.ndarray) -> Dict[str, float]:
    """
    Perform advanced data pattern analysis, computing various metrics
    including entropy, bit patterns, distribution, and more.

    Args:
        data (np.ndarray): The array of numeric values to analyze.

    Returns:
        Dict[str, float]: A dictionary of metrics related to the dataset.
    """
    metrics: Dict[str, float] = {}

    # Entropy analysis
    hist, _ = np.histogram(data, bins='auto')
    metrics['value_entropy'] = float(entropy(hist))

    # Bit pattern analysis
    bit_patterns = data.view(np.uint64)
    metrics['hamming_weight'] = float(np.sum([(x & (x - 1)) == 0 for x in bit_patterns]))

    # Distribution analysis
    mean_value = np.mean(data)
    median_value = np.median(data)
    metrics['skewness'] = float(np.abs(mean_value - median_value))
    std_dev = np.std(data)
    range_value = np.max(data) - np.min(data)
    metrics['range_ratio'] = float(range_value / std_dev) if std_dev != 0 else 0.0

    # Sequence analysis
    if len(data) > 1:
        diffs = np.diff(data)
        metrics['sequence_entropy'] = float(entropy(np.abs(diffs)))
    else:
        metrics['sequence_entropy'] = 0.0

    # Repetition analysis
    unique_vals = np.unique(data)
    metrics['uniqueness_ratio'] = float(len(unique_vals) / len(data)) if len(data) > 0 else 0.0

    return metrics

def optimize_parameters(metrics: Dict[str, float], success: bool) -> None:
    """
    Optimize sorting parameters based on performance metrics.

    Args:
        metrics (Dict[str, float]): The computed metrics from analyze_data_patterns.
        success (bool): A boolean indicating whether the sort iteration was successful.
    """
    global ADAPTIVE_PARAMS

    # Update success rate
    ADAPTIVE_PARAMS['total_runs'] += 1
    if success:
        ADAPTIVE_PARAMS['success_rate'] = (
            ADAPTIVE_PARAMS['success_rate'] * (ADAPTIVE_PARAMS['total_runs'] - 1) + 1
        ) / ADAPTIVE_PARAMS['total_runs']

    # Adjust parameters based on metrics
    if metrics['value_entropy'] > ADAPTIVE_PARAMS['entropy_threshold']:
        ADAPTIVE_PARAMS['entropy_threshold'] *= 1.1
    else:
        ADAPTIVE_PARAMS['entropy_threshold'] *= 0.9

    if metrics['uniqueness_ratio'] < 0.1:
        ADAPTIVE_PARAMS['bit_analysis_threshold'] *= 0.8

    if metrics['sequence_entropy'] < 2.0:
        ADAPTIVE_PARAMS['cluster_size_threshold'] *= 1.2

    # Persist updated parameters
    with open(params_file, 'w') as f:
        json.dump(ADAPTIVE_PARAMS, f)

################################################################################
# Enhanced EidosSort Algorithm
################################################################################

@jit(nopython=True, parallel=True, fastmath=True)
def parallel_quicksort(arr: np.ndarray, insertion_threshold: int) -> np.ndarray:
    """
    Numba-optimized parallel quicksort with enhanced pivot selection.

    Args:
        arr (np.ndarray): The array to be sorted in-place.
        insertion_threshold (int): The threshold below which insertion sort is used.

    Returns:
        np.ndarray: A sorted version of the input array.
    """
    if len(arr) <= 1:
        return arr
    if len(arr) <= insertion_threshold:
        return insertion_sort(arr)

    # Enhanced pivot selection using median of medians
    def get_pivot(array: np.ndarray) -> float:
        if len(array) <= 5:
            return float(np.median(array))
        chunks = array[:len(array) - len(array) % 5].reshape((-1, 5))
        medians = np.array([np.median(chunk) for chunk in chunks])
        return float(np.median(medians))

    pivot = get_pivot(arr)

    # Three-way partitioning for better handling of duplicates
    left_mask = arr < pivot
    right_mask = arr > pivot
    equal_mask = ~(left_mask | right_mask)

    left = arr[left_mask]
    middle = arr[equal_mask]
    right = arr[right_mask]

    # Parallel processing of partitions
    with objmode(left_sorted='float64[:]', right_sorted='float64[:]'):
        left_sorted = parallel_quicksort(left, insertion_threshold)
        right_sorted = parallel_quicksort(right, insertion_threshold)

    return np.concatenate((left_sorted, middle, right_sorted))

@jit(nopython=True, fastmath=True, cache=True)
def insertion_sort(arr: np.ndarray) -> np.ndarray:
    """
    Cache-optimized insertion sort for small arrays.

    Args:
        arr (np.ndarray): The array to be sorted in-place.

    Returns:
        np.ndarray: The same array, but sorted.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def eidos_sort(
    data: Union[list, np.ndarray],
    num_threads: Optional[int] = None,
    chunk_size: Optional[int] = None,
    use_clustering: bool = True,
    use_bit_analysis: bool = True,
    threshold: Optional[int] = None
) -> np.ndarray:
    """
    Enhanced EidosSort with improved optimizations and adaptive parameters.

    Args:
        data (Union[list, np.ndarray]): The data to be sorted. Will be converted to np.ndarray if not already.
        num_threads (Optional[int]): Number of parallel threads to utilize (defaults to heuristic-based).
        chunk_size (Optional[int]): Size of chunks for parallel processing.
        use_clustering (bool): Whether to apply clustering-based pre-sorting for low-uniqueness data.
        use_bit_analysis (bool): Whether to apply bit-pattern analysis to optimize sorting strategy.
        threshold (Optional[int]): Array size below which insertion sort is used (defaults to 16).

    Returns:
        np.ndarray: The sorted array.
    """
    start_time = time.perf_counter()

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.size <= 1:
        return data

    # Apply or default adaptive parameters
    chunk_size = chunk_size if chunk_size is not None else 1024
    threshold = threshold if threshold is not None else 16

    # Analyze data patterns
    metrics = analyze_data_patterns(data)

    # Optimize thread allocation heuristically
    if num_threads is None:
        cpu_count = psutil.cpu_count(logical=False)
        available_memory = psutil.virtual_memory().available
        # Use some fraction of CPU cores and memory-based limit
        num_threads = min(cpu_count * 2, max(2, available_memory // (data.nbytes * 2)))

    # Handle non-finite values (NaN, Inf)
    if not np.all(np.isfinite(data)):
        return _handle_special_values(data, num_threads, chunk_size)

    # Cache-based optimization
    cache_info = psutil.cpu_freq()
    if cache_info:
        l3_cache = 16 * 1024 * 1024  # Assume ~16 MB L3 cache as a heuristic
        chunk_size = min(chunk_size, l3_cache // data.itemsize)

    # Small array optimization
    if data.size <= threshold:
        return insertion_sort(data.copy())

    # Choose an optimal strategy based on data characteristics
    if metrics['value_entropy'] < ADAPTIVE_PARAMS['entropy_threshold']:
        if use_bit_analysis and data.size > ADAPTIVE_PARAMS['bit_analysis_threshold']:
            result = _analyze_and_sort_patterns(data, threshold)
            if result is not None:
                optimize_parameters(metrics, True)
                return result

    if metrics['uniqueness_ratio'] < 0.5:
        if use_clustering and data.size > ADAPTIVE_PARAMS['cluster_size_threshold']:
            result = _cluster_and_sort(data, num_threads, threshold)
            if result is not None:
                optimize_parameters(metrics, True)
                return result

    # Parallel sorting with chunk splitting
    if data.size > chunk_size:
        num_chunks = num_threads * 2
        chunks = np.array_split(data, num_chunks)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            sorted_chunks = list(executor.map(
                lambda chunk: parallel_quicksort(chunk, ADAPTIVE_PARAMS['insertion_threshold']),
                chunks
            ))
        result = parallel_merge(sorted_chunks)
        optimize_parameters(metrics, True)
        return result

    # Fallback to parallel quicksort
    result = parallel_quicksort(data, ADAPTIVE_PARAMS['insertion_threshold'])
    optimize_parameters(metrics, True)
    return result

def _handle_special_values(
    data: np.ndarray,
    num_threads: int,
    chunk_size: int
) -> np.ndarray:
    """
    Enhanced handling of NaN and Inf values by sorting the finite subset
    and reinserting infinities and NaNs in their proper positions.

    Args:
        data (np.ndarray): The original array containing special values.
        num_threads (int): The number of threads to utilize if further sorting is performed.
        chunk_size (int): Size of chunks for parallel operations if needed.

    Returns:
        np.ndarray: A sorted array with NaNs and infinities appropriately placed.
    """
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    finite_mask = np.isfinite(data)

    finite_data = data[finite_mask]
    sorted_finite = eidos_sort(finite_data, num_threads, chunk_size)

    result = np.full_like(data, np.nan)
    current_idx = 0

    # Place negative infinity first
    neg_inf = data[inf_mask & (data < 0)]
    if len(neg_inf) > 0:
        result[current_idx:current_idx + len(neg_inf)] = neg_inf
        current_idx += len(neg_inf)

    # Place sorted finite data
    result[current_idx:current_idx + len(sorted_finite)] = sorted_finite
    current_idx += len(sorted_finite)

    # Place positive infinity last
    pos_inf = data[inf_mask & (data > 0)]
    if len(pos_inf) > 0:
        result[current_idx:current_idx + len(pos_inf)] = pos_inf

    # NaNs remain at the end
    return result

def _analyze_and_sort_patterns(data: np.ndarray, threshold: int) -> Optional[np.ndarray]:
    """
    Enhanced bit-pattern analysis with adaptive thresholds for quicker sorting
    if the data's bit-level distribution is sufficiently constrained.

    Args:
        data (np.ndarray): The array of numeric values to analyze for bit patterns.
        threshold (int): The cutoff below which insertion sort is used.

    Returns:
        Optional[np.ndarray]: A sorted array if conditions are met, otherwise None.
    """
    bit_patterns = data.view(np.uint64)
    unique_patterns, counts = np.unique(bit_patterns >> 32, return_counts=True)
    bit_dist_entropy = -np.sum((counts / len(data)) * np.log2(counts / len(data)))

    # If bit-level distribution has low entropy, straightforward parallel quicksort might suffice
    if bit_dist_entropy < 6:
        return parallel_quicksort(data, ADAPTIVE_PARAMS['insertion_threshold'])
    return None

def _cluster_and_sort(
    data: np.ndarray,
    num_threads: int,
    threshold: int
) -> Optional[np.ndarray]:
    """
    Improved clustering-based approach where data is grouped by similarity (via KMeans),
    then each cluster is sorted individually and merged.

    Args:
        data (np.ndarray): The array of numeric values to cluster and sort.
        num_threads (int): The number of parallel threads to utilize in sorting.
        threshold (int): The cutoff below which insertion sort is used for small clusters.

    Returns:
        Optional[np.ndarray]: A sorted array if clustering succeeds, otherwise None.
    """
    try:
        n_clusters = min(int(np.sqrt(data.size) * 1.5), int(num_threads * 2))
        normalized_data = (data - np.mean(data)) / (np.std(data) + 1e-10)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init='auto',
            algorithm='elkan',
            max_iter=100
        )

        labels = kmeans.fit_predict(normalized_data.reshape(-1, 1))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(n_clusters):
                cluster_data = data[labels == i]
                if len(cluster_data) > threshold:
                    futures.append(
                        executor.submit(
                            parallel_quicksort, cluster_data, ADAPTIVE_PARAMS['insertion_threshold']
                        )
                    )
                else:
                    futures.append(executor.submit(insertion_sort, cluster_data))

            sorted_clusters = [f.result() for f in futures]

        return parallel_merge(sorted_clusters)
    except Exception as e:
        logger.warning(f"Clustering failed: {str(e)}, falling back to standard sort")
        return None

@jit(nopython=True, fastmath=True)
def merge_arrays(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Optimized merge function with SIMD vectorization for merging two sorted arrays.
    Equivalent to the merge step in merge sort.

    Args:
        left (np.ndarray): A sorted array.
        right (np.ndarray): A sorted array.

    Returns:
        np.ndarray: A merged sorted array containing all elements of left and right.
    """
    result = np.empty(len(left) + len(right), dtype=left.dtype)
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1

    if i < len(left):
        result[k:] = left[i:]
    elif j < len(right):
        result[k:] = right[j:]

    return result

def parallel_merge(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Merges multiple sorted arrays in parallel, combining them into a single
    sorted result.

    Args:
        arrays (List[np.ndarray]): List of sorted NumPy arrays.

    Returns:
        np.ndarray: A single merged and sorted NumPy array.
    """
    if len(arrays) <= 1:
        return arrays[0] if arrays else np.array([])

    if len(arrays) == 2:
        return merge_arrays(arrays[0], arrays[1])

    mid = len(arrays) // 2
    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(parallel_merge, arrays[:mid])
        right_future = executor.submit(parallel_merge, arrays[mid:])
        left_merged = left_future.result()
        right_merged = right_future.result()
        return merge_arrays(left_merged, right_merged)

################################################################################
# Benchmarking
################################################################################
def run_benchmarks() -> pd.DataFrame:
    """
    Enhanced benchmarking routine comparing EidosSort to NumPy's built-in sort (QuickSort).
    Iterates over increasing dataset sizes from 10^0 up to 10^6 (modifiable via max_exp).
    For each dataset size, collects performance metrics, correctness checks, and memory usage.

    Returns:
        pd.DataFrame: A DataFrame containing the complete benchmark results.
    """
    max_exp = 6  # Increase this to handle larger dataset sizes (up to 10^6 by default).
    benchmark_results: List[Dict[str, Union[str, float, bool]]] = []

    def calc_theoretical(n: int) -> Tuple[float, float, float]:
        """
        Compute theoretical complexities for reference: O(n), O(n log2(n)), and O(n^2).

        Args:
            n (int): Dataset size.

        Returns:
            Tuple[float, float, float]: The values for O(n), O(n log2(n)), and O(n^2).
        """
        return float(n), float(n * math.log2(n)), float(n * n)

    for i in track_progress(range(max_exp + 1), desc="Running benchmarks"):
        label = f"10^{i}"
        try:
            size = 10 ** i
            data = np.random.uniform(0, 1000, size=size)  # Uniform random dataset
            logger.info(f"Benchmarking dataset of size {label}...")

            o_n, o_nlogn, o_n2 = calc_theoretical(size)

            def run_benchmark(
                sort_func: Callable[[np.ndarray], np.ndarray],
                dataset: np.ndarray
            ) -> Tuple[float, np.ndarray]:
                """
                Time the provided sorting function on the given dataset over multiple runs
                to get stable average performance.

                Args:
                    sort_func (Callable): The sorting function to benchmark.
                    dataset (np.ndarray): The array to be sorted.

                Returns:
                    Tuple[float, np.ndarray]: (average runtime seconds, sorted array).

                Raises:
                    RuntimeError: If no valid benchmarking runs are completed.
                """
                times = []
                result = None

                # Pre-run warmup to mitigate JIT or first-run overhead (especially for Numba).
                try:
                    _ = sort_func(dataset.copy())
                except Exception as e:
                    logger.error(f"Error in warmup run: {str(e)}")
                    raise

                # Perform multiple benchmark runs to get an average timing.
                for _ in range(3):
                    try:
                        start = time.perf_counter()
                        output = sort_func(dataset.copy())
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                        result = output
                    except Exception as e:
                        logger.error(f"Error in benchmark run: {str(e)}")
                        raise

                if not times or result is None:
                    raise RuntimeError("No valid benchmark runs completed during sorting tests.")

                return float(np.mean(times)), result

            try:
                eidos_time, eidos_sorted = run_benchmark(eidos_sort, data)
                quick_time, quick_sorted = run_benchmark(np.sort, data)
            except Exception as e:
                logger.error(f"Error running benchmarks for size {label}: {str(e)}")
                continue

            # Verify correctness of EidosSort with a floating-point tolerance
            try:
                eidos_correct = np.allclose(eidos_sorted, quick_sorted, rtol=1e-10, atol=1e-10)
            except Exception as e:
                logger.error(f"Error verifying sort correctness for size {label}: {str(e)}")
                eidos_correct = False

            # Compute speedup factor
            try:
                speedup = quick_time / eidos_time if eidos_time > 0 else float('inf')
            except ZeroDivisionError:
                speedup = float('inf')
            except Exception as e:
                logger.error(f"Error calculating speedup for size {label}: {str(e)}")
                speedup = 0.0

            mem_usage_mb = float(round(data.nbytes / (1024 * 1024), 2))
            benchmark_results.append({
                "Dataset Size": str(label),
                "N": float(size),
                "O(n)": float(o_n),
                "O(nlogn)": float(o_nlogn),
                "O(n^2)": float(o_n2),
                "EidosSort Time (s)": float(round(eidos_time, 6)),
                "QuickSort Time (s)": float(round(quick_time, 6)),
                "Speedup vs QuickSort": float(round(speedup, 3)),
                "EidosSort Correct": bool(eidos_correct),
                "Memory Usage (MB)": mem_usage_mb,
                "Size_Numeric": float(size)
            })

            logger.info(f"Completed benchmark for dataset size {label} successfully.")

        except Exception as e:
            logger.error(f"Error benchmarking dataset {label}: {str(e)}")
            continue

    if not benchmark_results:
        raise RuntimeError("No benchmark results were collected. All benchmarking attempts failed.")

    df = pd.DataFrame(benchmark_results)

    required_columns = {
        "Dataset Size": str,
        "N": float,
        "O(n)": float,
        "O(nlogn)": float,
        "O(n^2)": float,
        "EidosSort Time (s)": float,
        "QuickSort Time (s)": float,
        "Speedup vs QuickSort": float,
        "EidosSort Correct": bool,
        "Memory Usage (MB)": float,
        "Size_Numeric": float
    }
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in the benchmark DataFrame: {missing_cols}")

    for col, dtype in required_columns.items():
        df[col] = df[col].astype(dtype)

    return df

################################################################################
# Metrics Module
################################################################################
def compute_additional_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an extended set of metrics for advanced performance analysis, including:
      - Efficiency and confidence intervals
      - Scalability and memory scores
      - Normalized "score" columns
      - Theoretical comparisons (Time vs O(nlogn), Memory vs Theoretical, etc.)

    Args:
        results (pd.DataFrame): DataFrame containing basic benchmark results.

    Returns:
        pd.DataFrame: An updated DataFrame containing additional metrics and columns.
    """
    df = results.copy()

    # Make sure we have numeric dataset sizes
    if "Size_Numeric" not in df.columns:
        df["Size_Numeric"] = df["Dataset Size"].apply(lambda x: float(10 ** int(x.split('^')[1])))

    # Basic efficiency metrics
    df["Efficiency"] = df["Speedup vs QuickSort"] / df["N"]
    df["Efficiency_CI"] = df["Efficiency"].std() * 1.96 / np.sqrt(len(df))

    # Scalability metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        df["Scalability"] = df["EidosSort Time (s)"] / df["N"]
        df["Scalability_Score"] = 1 - (df["Scalability"].pct_change() / df["N"].pct_change()).abs()

    # Memory efficiency
    theoretical_mem = df["N"] * 8 / (1024 * 1024)
    df["Memory_Efficiency"] = theoretical_mem / df["Memory Usage (MB)"]
    df["Memory_Overhead"] = df["Memory Usage (MB)"] - theoretical_mem

    with np.errstate(divide='ignore', invalid='ignore'):
        df["Memory_Score"] = 1 / (1 + df["Memory_Overhead"] / theoretical_mem)

    # Compare EidosSort time to O(nlogn)
    df["Time vs O(nlogn)"] = df["EidosSort Time (s)"] / df["O(nlogn)"]
    df["Time vs O(nlogn)_CI"] = df["Time vs O(nlogn)"].std() * 1.96 / np.sqrt(len(df))

    # Performance stability
    window = 3
    rolling_std = df["EidosSort Time (s)"].rolling(window=window, min_periods=1).std()
    rolling_mean = df["EidosSort Time (s)"].rolling(window=window, min_periods=1).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        df["Time_Consistency"] = rolling_std / rolling_mean

    # Growth rate analysis
    with np.errstate(divide='ignore', invalid='ignore'):
        df["Growth_Rate"] = df["EidosSort Time (s)"].pct_change() / df["N"].pct_change()
        df["Relative_Growth"] = df["Growth_Rate"] / (
            df["QuickSort Time (s)"].pct_change() / df["N"].pct_change()
        )

    # Combined efficiency score
    weights = {
        "speed": 0.4,
        "memory": 0.3,
        "stability": 0.3
    }
    df["Speed_Score"] = 1 / (1 + df["Time vs O(nlogn)"])
    df["Stability_Score"] = 1 - df["Time_Consistency"]  # Lower std => higher stability
    df["Overall_Efficiency"] = (
        weights["speed"] * df["Speed_Score"] +
        weights["memory"] * df["Memory_Score"] +
        weights["stability"] * df["Stability_Score"]
    )

    # Performance confidence intervals
    std_time = df["EidosSort Time (s)"].std() if "EidosSort Time (s)" in df.columns else None
    mean_time = df["EidosSort Time (s)"].mean() if "EidosSort Time (s)" in df.columns else None
    if std_time is not None and mean_time is not None:
        margin = std_time * 1.96 / np.sqrt(len(df))
        df["Performance_CI_Lower"] = mean_time - margin
        df["Performance_CI_Upper"] = mean_time + margin
    else:
        # Fallback handling
        col = "EidosSort Time (s)"
        if col in df.columns:
            col_std = df[col].std()
            col_mean = df[col].mean()
            margin = col_std * 1.96 / np.sqrt(len(df))
            df["Performance_CI_Lower"] = col_mean - margin
            df["Performance_CI_Upper"] = col_mean + margin
        else:
            df["Performance_CI_Lower"] = np.nan
            df["Performance_CI_Upper"] = np.nan

    # Normalize key score columns to 0–1
    for col in ["Speed_Score", "Memory_Score", "Stability_Score", "Overall_Efficiency"]:
        if col in df.columns:
            min_val, max_val = df[col].min(), df[col].max()
            if max_val == min_val:
                df[col] = 1.0
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)

    # Memory vs Theoretical
    df["Memory vs Theoretical"] = (
        df["Memory Usage (MB)"] / (df["N"] * 8.0 / (1024 * 1024))
    )

    required_columns = [
        'Dataset Size', 'N', 'O(n)', 'O(nlogn)', 'O(n^2)',
        'EidosSort Time (s)', 'QuickSort Time (s)', 'Speedup vs QuickSort',
        'EidosSort Correct', 'Memory Usage (MB)', 'Size_Numeric',
        'Time vs O(nlogn)', 'Memory vs Theoretical', 'Efficiency',
        'Efficiency_CI', 'Scalability', 'Scalability_Score',
        'Memory_Efficiency', 'Memory_Overhead', 'Memory_Score',
        'Time_Consistency', 'Growth_Rate', 'Relative_Growth',
        'Speed_Score', 'Stability_Score', 'Overall_Efficiency',
        'Performance_CI_Lower', 'Performance_CI_Upper'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in DataFrame after metrics computation: {missing_cols}")

    return df

################################################################################
# Visualization Module
################################################################################
def visualize_benchmark_results(results: pd.DataFrame) -> None:
    """
    Create and save comprehensive visualizations for the benchmark results with advanced analytics.
    Includes runtime analysis, speedup validation, memory usage, stability, efficiency, scalability,
    and crossover analyses. Also extends polynomial fits beyond the largest dataset to reveal
    potential future crossovers or performance behaviors.

    Args:
        results (pd.DataFrame): The DataFrame containing benchmark results.
    """
    # Make a safe copy
    results = results.copy()

    # Convert string sizes (e.g. "10^i") into numeric values if not done
    if "Size_Numeric" not in results.columns:
        results["Size_Numeric"] = results["Dataset Size"].apply(lambda x: float(10 ** int(x.split('^')[1])))

    # Determine extrapolation range: up to 2 orders of magnitude beyond largest dataset
    x_min = results["Size_Numeric"].min()
    x_max = results["Size_Numeric"].max()
    extrap_max = x_max * 100  # or 10x, 100x, etc.

    fig = plt.figure(figsize=(16, 40), constrained_layout=True)
    gs = fig.add_gridspec(8, 2)  # One extra row for an extended plot

    ############################################################################
    # 1. Runtime Analysis
    ############################################################################
    ax_runtime = fig.add_subplot(gs[0, :])
    ax_runtime.set_title("Algorithm Runtime Analysis with Statistical Fitting + Extrapolation", pad=20)
    ax_runtime.set_xlabel("Dataset Size (n)")
    ax_runtime.set_ylabel("Time (seconds)")
    ax_runtime.set_xscale('log')
    ax_runtime.set_yscale('log')
    ax_runtime.grid(True, which="both", ls="-", alpha=0.2)

    algorithms = [
        ("EidosSort Time (s)", "EidosSort", "blue"),
        ("QuickSort Time (s)", "QuickSort", "red")
    ]

    for time_col, algo_name, color in algorithms:
        ax_runtime.scatter(results["Size_Numeric"], results[time_col],
                           label=f"{algo_name} (Actual)", color=color, alpha=0.6)

        # Attempt multiple models: n log n, polynomial
        models = {
            'nlogn': lambda x, a, b: a * x * np.log2(x) + b,
            'polynomial': lambda x, a, b, c: a * (x**b) + c
        }
        best_r2 = -np.inf
        best_fit = None
        best_model_name = None

        x_data = results["Size_Numeric"].values
        y_data = results[time_col].values

        for model_name, model_func in models.items():
            try:
                popt, _ = curve_fit(model_func, x_data, y_data, maxfev=20000)
                y_pred = model_func(x_data, *popt)
                r2 = r2_score(y_data, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = (model_name, model_func, popt)
                    best_model_name = model_name
            except Exception:
                continue

        if best_fit is not None:
            model_name, model_func, popt = best_fit
            # Plot within measured range
            x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 200)
            y_smooth = model_func(x_smooth, *popt)
            ax_runtime.plot(x_smooth, y_smooth, '--', color=color,
                            label=f"{algo_name} Trend ({model_name}, R²={best_r2:.3f})")

            # Extrapolation beyond largest dataset size
            x_extrap = np.logspace(np.log10(x_max), np.log10(extrap_max), 200)
            y_extrap = model_func(x_extrap, *popt)
            ax_runtime.plot(x_extrap, y_extrap, ':', color=color, alpha=0.5,
                            label=f"{algo_name} Extrapolation ({model_name})")

    ax_runtime.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 2. Speedup Analysis with Confidence Interval
    ############################################################################
    ax_speedup = fig.add_subplot(gs[1, :])
    ax_speedup.set_title("Speedup Analysis with Statistical Validation + Polynomial Extrapolation", pad=20)
    ax_speedup.set_xlabel("Dataset Size (n)")
    ax_speedup.set_ylabel("Speedup Ratio")
    ax_speedup.set_xscale('log')
    ax_speedup.grid(True)

    speedup_data = results["Speedup vs QuickSort"].values
    ci = 0.95
    z_score = norm.ppf((1 + ci) / 2)

    std_err = speedup_data.std() / np.sqrt(len(speedup_data))
    ci_lower = speedup_data.mean() - z_score * std_err
    ci_upper = speedup_data.mean() + z_score * std_err

    ax_speedup.plot(results["Size_Numeric"], speedup_data, 'g-o', label="Speedup Ratio")
    ax_speedup.fill_between(results["Size_Numeric"], ci_lower, ci_upper,
                            color='g', alpha=0.2,
                            label=f"{int(ci*100)}% Confidence Interval")

    # Fit polynomial in log space for speedup
    try:
        z_poly = np.polyfit(np.log10(results["Size_Numeric"]), speedup_data, 2)
        p_poly = np.poly1d(z_poly)
        x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        y_smooth = p_poly(np.log10(x_smooth))
        r2_speedup = r2_score(speedup_data, p_poly(np.log10(results["Size_Numeric"])))
        ax_speedup.plot(x_smooth, y_smooth, 'r--', label=f"Trend (R²={r2_speedup:.3f})")

        # Extrapolate speedup beyond largest dataset
        x_extrap = np.logspace(np.log10(x_max), np.log10(extrap_max), 200)
        y_extrap = p_poly(np.log10(x_extrap))
        ax_speedup.plot(x_extrap, y_extrap, 'r:', alpha=0.5, label="Speedup Extrapolation")
    except Exception as ex:
        logger.warning(f"Unable to fit polynomial for speedup trend: {str(ex)}")

    ax_speedup.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 3. Memory Efficiency Analysis
    ############################################################################
    ax_memory = fig.add_subplot(gs[2, :])
    ax_memory.set_title("Memory Usage Analysis with Efficiency Metrics", pad=20)
    ax_memory.set_xlabel("Dataset Size (n)")
    ax_memory.set_ylabel("Memory Usage (MB)")
    ax_memory.set_xscale('log')
    ax_memory.set_yscale('log')
    ax_memory.grid(True)

    actual_mem = results["Memory Usage (MB)"].values
    theoretical_mem = results["Size_Numeric"].values * 8 / (1024 * 1024)

    ax_memory.plot(results["Size_Numeric"], actual_mem, 'b-o', label="Actual Memory Usage")
    ax_memory.plot(results["Size_Numeric"], theoretical_mem, 'r--', label="Theoretical Minimum")

    overhead = actual_mem - theoretical_mem
    ax_memory.fill_between(results["Size_Numeric"], theoretical_mem, actual_mem,
                           alpha=0.3, color='gray', label="Memory Overhead")

    meff_col = "Memory_Efficiency"
    if meff_col in results.columns:
        efficiency_ratio = results[meff_col].mean()
        ax_memory.text(
            0.02, 0.98,
            f"Avg Mem Efficiency: {efficiency_ratio:.2f}",
            transform=ax_memory.transAxes,
            verticalalignment='top'
        )
    ax_memory.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 4. Performance Stability Analysis
    ############################################################################
    ax_stability = fig.add_subplot(gs[3, :])
    ax_stability.set_title("Performance Stability Analysis", pad=20)
    ax_stability.set_xlabel("Dataset Size (n)")
    ax_stability.set_ylabel("Execution Time (s)")
    ax_stability.set_xscale('log')
    ax_stability.grid(True)

    window = 3
    rolling_mean = results["EidosSort Time (s)"].rolling(window=window, min_periods=1).mean()
    rolling_std = results["EidosSort Time (s)"].rolling(window=window, min_periods=1).std()

    ax_stability.plot(results["Size_Numeric"], rolling_mean, 'b-o', label="Rolling Mean (Window=3)")
    ax_stability.fill_between(
        results["Size_Numeric"],
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.3, color='blue', label="±1 Std Dev"
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = (rolling_std / rolling_mean).mean()

    ax_stability.text(
        0.02, 0.98,
        f"Coefficient of Variation: {cv:.3f}",
        transform=ax_stability.transAxes,
        verticalalignment='top'
    )
    ax_stability.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 5. Algorithmic Efficiency Metrics
    ############################################################################
    ax_efficiency = fig.add_subplot(gs[4, :])
    ax_efficiency.set_title("Comprehensive Efficiency Metrics", pad=20)
    ax_efficiency.set_xlabel("Dataset Size (n)")
    ax_efficiency.set_ylabel("Efficiency Metric")
    ax_efficiency.set_xscale('log')
    ax_efficiency.grid(True)

    # Possible columns from compute_additional_metrics
    metric_plots = [
        ("Time vs O(nlogn)", "purple", "Time Efficiency"),
        ("Memory vs Theoretical", "orange", "Memory Efficiency"),
        ("Efficiency", "green", "Overall Efficiency")
    ]
    for col, color, label in metric_plots:
        if col in results.columns:
            ax_efficiency.plot(
                results["Size_Numeric"], results[col],
                color=color, marker='o', label=label
            )

    ax_efficiency.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 6. Scalability Analysis
    ############################################################################
    ax_scale = fig.add_subplot(gs[5, :])
    ax_scale.set_title("Scalability Analysis", pad=20)
    ax_scale.set_xlabel("Dataset Size (n)")
    ax_scale.set_ylabel("Scalability Factor (ΔTime/Δn)")
    ax_scale.set_xscale('log')
    ax_scale.grid(True)

    with np.errstate(divide='ignore', invalid='ignore'):
        scalability = results["EidosSort Time (s)"].pct_change() / results["Size_Numeric"].pct_change()

    ax_scale.plot(results["Size_Numeric"][1:], scalability[1:], 'r-o', label="Scalability Factor")
    ax_scale.axhline(y=1, color='g', linestyle='--', label="Ideal Linear Scaling")

    avg_scalability = np.nanmean(scalability)
    ax_scale.text(
        0.02, 0.98,
        f"Average Scalability Factor: {avg_scalability:.3f}",
        transform=ax_scale.transAxes,
        verticalalignment='top'
    )
    ax_scale.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 7. Crossover Analysis
    ############################################################################
    ax_crossover = fig.add_subplot(gs[6, :])
    ax_crossover.set_title("Algorithm Crossover Analysis", pad=20)
    ax_crossover.set_xlabel("Dataset Size (n)")
    ax_crossover.set_ylabel("Relative Efficiency (QuickSort/EidosSort)")
    ax_crossover.set_xscale('log')
    ax_crossover.grid(True)

    relative_efficiency = results["QuickSort Time (s)"] / results["EidosSort Time (s)"]
    ax_crossover.plot(results["Size_Numeric"], relative_efficiency, 'b-o', label="QSort/EidosSort Ratio")
    ax_crossover.axhline(y=1, color='r', linestyle='--', label='Efficiency Threshold')

    # Find crossovers
    diffs = np.signbit(relative_efficiency - 1)
    crossover_indices = np.where(np.diff(diffs))[0]
    for idx in crossover_indices:
        crossover_x = results["Size_Numeric"].iloc[idx]
        ax_crossover.axvline(x=crossover_x, color='g', linestyle=':', alpha=0.5)
        ax_crossover.text(crossover_x, 0.5,
                          f'Crossover @ n={crossover_x:.0f}', rotation=90)

    # Attempt polynomial fits for each algorithm to see if future crossovers might appear
    # Fit the ratio directly or the difference
    x_data = results["Size_Numeric"].values
    y_ratio = relative_efficiency.values

    try:
        # Fit polynomial in log-log space for the ratio
        z_ratio = np.polyfit(np.log10(x_data), np.log10(y_ratio), 2)
        p_ratio = np.poly1d(z_ratio)
        x_extrap_c = np.logspace(np.log10(x_max), np.log10(extrap_max), 200)
        # Reconstruct ratio from log10 fit => ratio = 10^(p_ratio(log10(x)))
        ratio_extrap = 10 ** p_ratio(np.log10(x_extrap_c))

        ax_crossover.plot(x_extrap_c, ratio_extrap, 'm:', alpha=0.7,
                          label="Future Ratio Extrapolation")

        # Check for intersection with ratio=1
        # Solve 10^(p_ratio(log10(x))) = 1 => p_ratio(log10(x)) = 0 => log10(x) = r
        # That means we solve p_ratio(r) = 0 for r, then x = 10^r
        roots = (p_ratio - 0).roots
        for r in roots:
            # We only consider real solutions in a plausible domain
            if np.isreal(r):
                x_root = 10 ** r
                if x_root > x_max and x_root < extrap_max:
                    ax_crossover.axvline(x=x_root, color='m', linestyle='--', alpha=0.6)
                    ax_crossover.text(x_root, 1.2,
                                      f"Future Crossover ~ {x_root:.1f}",
                                      rotation=90, color='magenta')
    except Exception as ex:
        logger.warning(f"Unable to perform future ratio extrapolation: {str(ex)}")

    ax_crossover.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ############################################################################
    # 8. Extra Subplot or Additional Info (Optional)
    ############################################################################
    ax_info = fig.add_subplot(gs[7, :])
    ax_info.axis("off")
    ax_info.set_title("Additional Observations & Notes", pad=20, fontsize=12)
    txt = (
        "• Charts now include polynomial extrapolations well beyond the largest dataset.\n"
        "• Check future crossovers: if EidosSort might outperform QuickSort at bigger N.\n"
        "• Speedup ratio and ratio-based crossovers extended with log-polynomial fits.\n"
        "• Use these extended analyses for capacity planning or algorithmic scaling insights.\n"
    )
    ax_info.text(0, 0.9, txt, fontsize=10, va='top')

    # Save figure
    plt.savefig(output_dir / "benchmark_visualization.png",
                dpi=300, bbox_inches='tight',
                metadata={"Creator": "EidosSort Benchmark Suite"})
    plt.close()

################################################################################
# Analysis Module
################################################################################
def analyze_sorting_performance(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis and performance evaluation with
    detailed insights and numeric verifications. Generates a dictionary containing:
      - Basic stats (mean speedup, confidence interval, memory efficiency).
      - Identifies thresholds for significant speedup or memory concerns.
      - Estimates complexity via logarithmic fitting (EidosSort vs. QuickSort).
      - Builds a polynomial model for performance prediction.
      - Provides optimization recommendations based on results.
      - Assesses stability with a numeric score.

    Args:
        results (pd.DataFrame): DataFrame with all benchmark and metric columns.

    Returns:
        Dict[str, Any]: Analysis dictionary with stats, complexities, recommended improvements, etc.
    """
    analysis = {}
    df = results.copy()

    if "Size_Numeric" not in df.columns:
        df["Size_Numeric"] = df["Dataset Size"].apply(lambda x: float(10 ** int(x.split('^')[1])))

    mean_speedup = float(df["Speedup vs QuickSort"].mean())
    speedup_se = stats.sem(df["Speedup vs QuickSort"])
    speedup_ci = t.interval(0.95, len(df) - 1, loc=mean_speedup, scale=speedup_se)

    # Average memory usage vs theoretical
    avg_memory_usage = (df["Memory Usage (MB)"] / (df["Size_Numeric"] * 8 / (1024 * 1024))).mean()

    analysis["basic_stats"] = {
        "mean_speedup": mean_speedup,
        "speedup_ci": speedup_ci,
        "memory_efficiency": float(avg_memory_usage)
    }

    # Threshold-based observations
    threshold_speedup = mean_speedup + df["Speedup vs QuickSort"].std()
    threshold_memory = df["Memory Usage (MB)"].mean() * 1.5
    high_speedups = df[df["Speedup vs QuickSort"] > threshold_speedup]["Dataset Size"].tolist()
    memory_concerns = df[df["Memory Usage (MB)"] > threshold_memory]["Dataset Size"].tolist()

    analysis["thresholds"] = {
        "significant_speedup": high_speedups,
        "memory_concern": memory_concerns
    }

    # Complexity analysis
    eidos_col = df["EidosSort Time (s)"].replace(0, np.nan)
    quick_col = df["QuickSort Time (s)"].replace(0, np.nan)
    size_col = df["Size_Numeric"]

    valid_mask_eidos = ~np.isnan(eidos_col) & (eidos_col > 0) & (size_col > 0)
    valid_mask_quick = ~np.isnan(quick_col) & (quick_col > 0) & (size_col > 0)

    def fit_loglog(x, y) -> float:
        x_log = np.log10(x)
        y_log = np.log10(y)
        slope, intercept = np.polyfit(x_log, y_log, 1)
        return slope

    if valid_mask_eidos.sum() > 1:
        eidos_slope = float(fit_loglog(size_col[valid_mask_eidos], eidos_col[valid_mask_eidos]))
    else:
        eidos_slope = 1.0

    if valid_mask_quick.sum() > 1:
        quick_slope = float(fit_loglog(size_col[valid_mask_quick], quick_col[valid_mask_quick]))
    else:
        quick_slope = 1.0

    analysis["complexity"] = {
        "eidos_complexity": f"O(n^{eidos_slope:.3f})",
        "quick_complexity": f"O(n^{quick_slope:.3f})",
        "relative_complexity": eidos_slope / quick_slope if quick_slope != 0 else float('inf')
    }

    # Performance prediction model
    valid_mask_both = valid_mask_eidos & valid_mask_quick
    X = size_col[valid_mask_both].to_numpy().reshape(-1, 1)
    y = eidos_col[valid_mask_both].to_numpy()

    if len(X) > 1:
        model = make_pipeline(
            PolynomialFeatures(degree=2),
            StandardScaler(),
            LinearRegression()
        )
        model.fit(X, y)
        analysis["prediction_model"] = model
    else:
        analysis["prediction_model"] = None

    # Optimization recommendations
    analysis["recommendations"] = []
    if analysis["complexity"]["relative_complexity"] > 1.1:
        analysis["recommendations"].append(
            "Investigate pivot selection or advanced partitioning to reduce complexity."
        )

    if analysis["basic_stats"]["memory_efficiency"] > 1.2:
        analysis["recommendations"].append(
            "Explore memory usage optimizations or partitioning strategies."
        )

    # Performance stability
    eidos_std = df["EidosSort Time (s)"].std()
    eidos_mean = df["EidosSort Time (s)"].mean()
    if eidos_mean > 0:
        stability_score = 1 - (eidos_std / eidos_mean)
    else:
        stability_score = 0.0

    analysis["stability"] = {
        "score": stability_score,
        "assessment": "Stable" if stability_score > 0.9 else "Needs Improvement"
    }

    # Save textual report
    with open(output_dir / "performance_analysis.txt", 'w') as f:
        f.write("Advanced Performance Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. Statistical Summary\n")
        f.write(f" - Mean Speedup: {analysis['basic_stats']['mean_speedup']:.3f}\n")
        f.write(f" - 95% CI for Speedup: {analysis['basic_stats']['speedup_ci']}\n")
        f.write(f" - Memory Efficiency (avg): {analysis['basic_stats']['memory_efficiency']:.3f}\n\n")
        f.write("2. Complexity Analysis\n")
        f.write(f" - EidosSort: {analysis['complexity']['eidos_complexity']}\n")
        f.write(f" - QuickSort: {analysis['complexity']['quick_complexity']}\n")
        f.write(f" - Relative Scaling: {analysis['complexity']['relative_complexity']:.3f}\n\n")
        f.write("3. Performance Stability\n")
        f.write(f" - Stability Score: {analysis['stability']['score']:.3f}\n")
        f.write(f" - Assessment: {analysis['stability']['assessment']}\n\n")
        f.write("4. Optimization Recommendations\n")
        if analysis["recommendations"]:
            for idx, rec in enumerate(analysis["recommendations"], start=1):
                f.write(f"   {idx}. {rec}\n")
        else:
            f.write("   None\n")

    return analysis

################################################################################
# Main Execution
################################################################################
def main():
    """
    Main execution function with robust error handling, logging, and performance tracking.
    Orchestrates the entire benchmark suite:
      - Runs benchmarks with track_progress
      - Computes additional metrics
      - Saves results to disk
      - Generates visualizations
      - Performs deeper analyses and logs a final summary.
    """
    logger.info("Initializing EidosSort benchmark suite...")
    start_time = time.perf_counter()

    try:
        # Step 1: Run benchmarks
        results = run_benchmarks()
        logger.info("All benchmarks completed successfully.")

        # Step 2: Compute additional metrics
        results = compute_additional_metrics(results)
        logger.info("Additional metrics computed and validated.")

        # Step 3: Save results & metadata
        results.to_csv(output_dir / "benchmark_results.csv", index=False)
        metadata = {
            "timestamp": str(pd.Timestamp.utcnow()),
            "version": "2.0.0",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        with open(output_dir / "benchmark_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Step 4: Generate visualizations
        visualize_benchmark_results(results)
        logger.info("Benchmark visualizations generated successfully.")

        # Step 5: Perform deeper analysis
        analysis = analyze_sorting_performance(results)
        logger.info("Performance analysis completed.")

        # Print quick summary
        print("\nBenchmark Suite Summary:")
        print("=" * 80)
        print(f"Total Datasets: {len(results)}")
        avg_speedup = results['Speedup vs QuickSort'].mean()
        print(f"Average Speedup vs QuickSort: {avg_speedup:.2f}x")

        if "Memory_Efficiency" in results.columns:
            mem_efficiency = results["Memory_Efficiency"].mean()
            print(f"Average Memory Efficiency: {mem_efficiency:.2f}")

        if "Overall_Efficiency" in results.columns:
            overall_perf = results["Overall_Efficiency"].mean()
            print(f"Overall Performance Score: {overall_perf:.2f}")

        print("=" * 80)

        total_time = time.perf_counter() - start_time
        logger.info(f"Benchmark suite completed in {total_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Critical error in benchmark suite: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
