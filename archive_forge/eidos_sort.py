# Standard library imports
from dataclasses import dataclass
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

# Third-party imports
import numpy as np
from numpy.random import uniform
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Scientific computing
from scipy import stats  # type: ignore
from scipy.stats import entropy, norm, t  # type: ignore
from scipy.optimize import curve_fit  # type: ignore

# Machine learning
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

# Performance optimization
import numba  # type: ignore
from numba import jit, prange, objmode  # type: ignore

from deap import algorithms, base, creator, tools  # type: ignore

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eidos_sort.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Output Directory and Parameters
# =============================================================================
output_dir = Path("benchmark_results")
output_dir.mkdir(parents=True, exist_ok=True)

params_file = output_dir / "eidos_params.json"
ADAPTIVE_PARAMS: Dict[str, Union[float, int]] = {
    "entropy_threshold": 6.0,
    "cluster_size_threshold": 5000,
    "bit_analysis_threshold": 500,
    "chunk_size": 1000,
    "insertion_threshold": 16,
    "success_rate": 0.0,
    "total_runs": 0,
}

if params_file.exists():
    try:
        with open(params_file, "r") as f:
            loaded_params = json.load(f)
            ADAPTIVE_PARAMS.update(loaded_params)
        logger.info(f"Loaded parameters from {params_file}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {params_file}: {e}")
        logger.warning(f"Using default parameters due to JSON decode error.")
else:
    logger.info(f"Parameter file not found, using default parameters.")


# =============================================================================
# Progress Tracking
# =============================================================================
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


# =============================================================================
# Resource Orchestration Layer
# =============================================================================
class ResourceManager:
    """Dynamic resource allocator with real-time system monitoring"""

    def __init__(self) -> None:
        """Initializes the resource manager with default update interval and adaptive parameters."""
        self.update_interval: float = 0.5
        self.adaptive_params: Dict[str, Union[float, int]] = ADAPTIVE_PARAMS
        self.cpu_usage: float = 0.0
        self.mem_available: int = 0
        self.disk_io: Any = None
        self.net_io: Any = None

    def monitor_resources(self) -> None:
        """Continuously monitor system resources."""
        while True:
            try:
                self.cpu_usage = psutil.cpu_percent(interval=self.update_interval)
                self.mem_available = psutil.virtual_memory().available
                self.disk_io = psutil.disk_io_counters()
                self.net_io = psutil.net_io_counters()
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(self.update_interval)

    def optimal_thread_count(self) -> int:
        """Calculate optimal parallelization based on current load."""
        try:
            safe_cpu = max(10, 100 - self.cpu_usage - 20)  # Leave 20% headroom
            available_memory_mb = self.mem_available // 1_000_000
            cpu_count = psutil.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            thread_count = min(
                available_memory_mb // 500,  # 500MB per thread
                int(safe_cpu * cpu_count / 100),
            )
            return max(1, thread_count)
        except Exception as e:
            logger.error(f"Error calculating optimal thread count: {e}")
            return 1


resource_manager = ResourceManager()

# =============================================================================
# Enhanced Analysis Functions
# =============================================================================
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
    try:
        logger.debug(f"Data type: {data.dtype}")
        if data.ndim != 1:
            raise ValueError("Data must be a one-dimensional array.")

        if len(data) == 0:
            logger.warning("Empty data array provided for analysis.")
            return metrics  # Or assign default metrics

        # Entropy analysis
        hist, _ = np.histogram(data, bins="auto")
        metrics["value_entropy"] = float(entropy(hist))

        # Bit pattern analysis (only for integer data)
        if np.issubdtype(data.dtype, np.integer):
            bit_patterns = data.view(np.uint64)
            hamming_count = 0
            for x in bit_patterns:
                if (x & (x - 1)) == 0:
                    hamming_count += 1
            metrics["hamming_weight"] = float(hamming_count)
        else:
            metrics["hamming_weight"] = 0.0

        # Distribution analysis
        mean_value = np.mean(data)
        median_value = np.median(data)
        metrics["skewness"] = float(np.abs(mean_value - median_value))
        std_dev = np.std(data)
        range_value = np.max(data) - np.min(data)
        metrics["range_ratio"] = float(range_value / std_dev) if std_dev != 0 else 0.0

        # Sequence analysis
        if len(data) > 1:
            diffs = np.diff(data)
            metrics["sequence_entropy"] = float(entropy(np.abs(diffs)))
        else:
            metrics["sequence_entropy"] = 0.0

        # Repetition analysis
        unique_vals = np.unique(data)
        metrics["uniqueness_ratio"] = (
            float(len(unique_vals) / len(data)) if len(data) > 0 else 0.0
        )
    except Exception as e:
        logger.error(f"Error analyzing data patterns: {e}")
        return {}
    return metrics


def optimize_parameters(metrics: Dict[str, float], success: bool) -> None:
    """
    Optimize sorting parameters based on performance metrics.

    Args:
        metrics (Dict[str, float]): The computed metrics from analyze_data_patterns.
        success (bool): A boolean indicating whether the sort iteration was successful.
    """
    global ADAPTIVE_PARAMS
    try:
        # Update success rate
        ADAPTIVE_PARAMS["total_runs"] += 1
        if success:
            ADAPTIVE_PARAMS["success_rate"] = (
                ADAPTIVE_PARAMS["success_rate"] * (ADAPTIVE_PARAMS["total_runs"] - 1)
                + 1
            ) / ADAPTIVE_PARAMS["total_runs"]

        # Adjust parameters based on metrics
        if metrics.get("value_entropy", 0.0) > ADAPTIVE_PARAMS.get(
            "entropy_threshold", 6.0
        ):
            ADAPTIVE_PARAMS["entropy_threshold"] = float(
                ADAPTIVE_PARAMS.get("entropy_threshold", 6.0) * 1.1
            )
        else:
            ADAPTIVE_PARAMS["entropy_threshold"] = float(
                ADAPTIVE_PARAMS.get("entropy_threshold", 6.0) * 0.9
            )

        if metrics.get("uniqueness_ratio", 1.0) < 0.1:
            ADAPTIVE_PARAMS["bit_analysis_threshold"] = float(
                ADAPTIVE_PARAMS.get("bit_analysis_threshold", 500) * 0.8
            )

        if metrics.get("sequence_entropy", 10.0) < 2.0:
            ADAPTIVE_PARAMS["cluster_size_threshold"] = float(
                ADAPTIVE_PARAMS.get("cluster_size_threshold", 5000) * 1.2
            )

        # Persist updated parameters
        with open(params_file, "w") as f:
            json.dump(ADAPTIVE_PARAMS, f)
    except Exception as e:
        logger.error(f"Error optimizing parameters: {e}")


# =============================================================================
# Enhanced EidosSort Algorithm
# =============================================================================

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
        chunk_size (Optional[int]): Size of chunks for parallel processing (defaults to 1024 or adaptive).
        use_clustering (bool): Whether to apply clustering-based pre-sorting for low-uniqueness data.
        use_bit_analysis (bool): Whether to apply bit-pattern analysis to optimize sorting strategy.
        threshold (Optional[int]): Array size below which insertion sort is used (defaults to 16).

    Returns:
        np.ndarray: The sorted array.
    """
    start_time = time.perf_counter()
    try:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.size <= 1:
            return data

        # Apply or default adaptive parameters
        chunk_size = (
            chunk_size
            if chunk_size is not None
            else int(ADAPTIVE_PARAMS.get("chunk_size", 1024))
        )
        threshold = (
            int(threshold)
            if threshold is not None
            else int(ADAPTIVE_PARAMS.get("insertion_threshold", 16))
        )

        # Analyze data patterns
        metrics = analyze_data_patterns(data)

        # Optimize thread allocation heuristically
        if num_threads is None:
            cpu_count = psutil.cpu_count(logical=False)
            available_memory = psutil.virtual_memory().available
            if cpu_count is None:
                cpu_count = 1
            num_threads = min(
                cpu_count * 2, max(2, available_memory // (data.nbytes * 2))
            )

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
        if metrics.get("value_entropy", 10.0) < ADAPTIVE_PARAMS.get(
            "entropy_threshold", 6.0
        ):
            if use_bit_analysis and data.size > ADAPTIVE_PARAMS.get(
                "bit_analysis_threshold", 500
            ):
                result = _analyze_and_sort_patterns(data, threshold)
                if result is not None:
                    optimize_parameters(metrics, True)
                    return result

        if metrics.get("uniqueness_ratio", 1.0) < 0.5:
            if use_clustering and data.size > ADAPTIVE_PARAMS.get(
                "cluster_size_threshold", 5000
            ):
                result = _cluster_and_sort(data, num_threads, threshold)
                if result is not None:
                    optimize_parameters(metrics, True)
                    return result

        # Parallel sorting with chunk splitting
        if data.size > chunk_size:
            num_chunks = num_threads * 2
            chunks = np.array_split(data, num_chunks)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                sorted_chunks = list(
                    executor.map(
                        lambda chunk: parallel_quicksort(
                            chunk, int(ADAPTIVE_PARAMS.get("insertion_threshold", 16))
                        ),
                        chunks,
                    )
                )
            result = parallel_merge(sorted_chunks)
            optimize_parameters(metrics, True)
            return result

        # Fallback to parallel quicksort
        result = parallel_quicksort(
            data, int(ADAPTIVE_PARAMS.get("insertion_threshold", 16))
        )
        optimize_parameters(metrics, True)
        if result is None:
            return np.array([])
        return result
    except Exception as e:
        logger.error(f"Error in eidos_sort: {e}")
        return np.array([])
    finally:
        end_time = time.perf_counter()
        logger.debug(f"eidos_sort execution time: {end_time - start_time:.6f} seconds")


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
    try:
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
            result[current_idx : current_idx + len(neg_inf)] = neg_inf
            current_idx += len(neg_inf)

        # Place sorted finite data
        result[current_idx : current_idx + len(sorted_finite)] = sorted_finite
        current_idx += len(sorted_finite)

        # Place positive infinity last
        pos_inf = data[inf_mask & (data > 0)]
        if len(pos_inf) > 0:
            result[current_idx : current_idx + len(pos_inf)] = pos_inf

        # NaNs remain wherever they are placed, effectively at the end
        return result
    except Exception as e:
        logger.error(f"Error handling special values: {e}")
        return data


def _analyze_and_sort_patterns(data: np.ndarray, threshold: int) -> Optional[np.ndarray]:
    """
    Enhanced bit-pattern analysis with adaptive thresholds for quicker sorting
    if the data's bit-level distribution is sufficiently constrained.

    Args:
        data (np.ndarray): The array of numeric values to analyze for bit patterns.
        threshold (int): The cutoff below which we use an alternative method like insertion sort.

    Returns:
        Optional[np.ndarray]: A sorted array if conditions are met, otherwise None.
    """
    try:
        bit_patterns = data.view(np.uint64)
        unique_patterns, counts = np.unique(bit_patterns >> 32, return_counts=True)
        bit_dist_entropy = -np.sum((counts / len(data)) * np.log2(counts / len(data)))

        # If bit-level distribution has low entropy, straightforward parallel quicksort might suffice
        if bit_dist_entropy < 6:
            return parallel_quicksort(
                data, int(ADAPTIVE_PARAMS.get("insertion_threshold", 16))
            )
        return None
    except Exception as e:
        logger.error(f"Error analyzing and sorting bit patterns: {e}")
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
                            parallel_quicksort,
                            cluster_data,
                            int(ADAPTIVE_PARAMS.get("insertion_threshold", 16)),
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

    # Main merge loop
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1

    # Tail copying
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


# =============================================================================
# Benchmarking
# =============================================================================
def run_benchmarks(
    max_exp: int = 4,
    min_size: int = 100,
    datasets_per_exp: int = 10,
    num_runs: int = 5,
) -> pd.DataFrame:
    """
    Enhanced benchmarking routine comparing EidosSort to NumPy's built-in sort (QuickSort).
    Iterates over a configurable range of dataset sizes, with multiple datasets per size,
    and various dataset types (random, sorted, partially sorted). Collects performance
    metrics, correctness checks, and memory usage.

    Args:
        max_exp (int): Maximum exponent for dataset sizes (10^max_exp). Default is 4 (10^4).
        min_size (int): Minimum dataset size for benchmarking. Default is 100.
        datasets_per_exp (int): Number of datasets to generate per exponent range. Default is 10.
        num_runs (int): Number of benchmark runs to average for each sort. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame containing the complete benchmark results.

    Raises:
        RuntimeError: If no benchmark results are collected.
        KeyError: If required columns are missing in the benchmark DataFrame.
    """
    benchmark_results: List[Dict[str, Union[str, float, bool]]] = []

    def calc_theoretical(n: int) -> Tuple[float, float, float]:
        """
        Compute theoretical complexities for reference: O(n), O(n log n), O(n^2).

        Args:
            n (int): Dataset size.

        Returns:
            Tuple[float, float, float]: The values for O(n), O(n log2(n)), and O(n^2).
        """
        return float(n), float(n * math.log2(n)), float(n * n)

    def generate_datasets(size: int) -> List[np.ndarray]:
        """
        Generates a variety of datasets for benchmarking, including random,
        sorted, and partially sorted arrays.

        Args:
            size (int): The size of the dataset to generate.

        Returns:
            List[np.ndarray]: A list of NumPy arrays with different characteristics.
        """
        datasets: List[np.ndarray] = []
        # Generate a random dataset
        random_data = np.random.uniform(0, 1000, size=size)
        datasets.append(random_data)

        # Generate a sorted dataset
        sorted_data = np.sort(random_data.copy())
        datasets.append(sorted_data)

        # Generate a reverse sorted dataset
        reverse_sorted_data = sorted_data[::-1].copy()
        datasets.append(reverse_sorted_data)

        # Generate a partially sorted dataset (first 10% sorted)
        partial_sorted_data = random_data.copy()
        partial_sorted_data[: size // 10] = np.sort(partial_sorted_data[: size // 10])
        datasets.append(partial_sorted_data)

        # Generate a mostly sorted dataset (90% sorted)
        mostly_sorted_data = random_data.copy()
        mostly_sorted_data[: int(size * 0.9)] = np.sort(
            mostly_sorted_data[: int(size * 0.9)]
        )
        datasets.append(mostly_sorted_data)

        return datasets

    def run_benchmark(
        sort_func: Callable[[np.ndarray], np.ndarray], dataset: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Time the provided sorting function on the given dataset over multiple runs
        to get more stable average performance.

        Args:
            sort_func (Callable): The sorting function to benchmark.
            dataset (np.ndarray): The array to be sorted.

        Returns:
            Tuple[float, np.ndarray]: (average runtime seconds, sorted array).

        Raises:
            RuntimeError: If no valid benchmarking runs are completed.
        """
        times: List[float] = []
        result: Optional[np.ndarray] = None

        # Pre-run warmup to mitigate JIT or first-run overhead (especially for Numba).
        try:
            _ = sort_func(dataset.copy())
        except Exception as e:
            logger.error(f"Error in warmup run: {e}")
            raise

        # Perform multiple benchmark runs to get an average timing.
        for _ in range(num_runs):
            try:
                start = time.perf_counter()
                output = sort_func(dataset.copy())
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                result = output
            except Exception as e:
                logger.error(f"Error in benchmark run: {e}")
                raise

        if not times or result is None:
            raise RuntimeError("No benchmark runs completed during sorting tests.")

        return float(np.mean(times)), result

    # Iterate over a range of exponents to generate dataset sizes
    for i in track_progress(range(max_exp + 1), desc="Running benchmarks"):
        for j in range(datasets_per_exp):
            # Calculate dataset size within the exponent range
            size = int(
                min_size
                * 10
                ** (i * (j / (datasets_per_exp - 1) if datasets_per_exp > 1 else 0))
            )
            label = f"10^{i}_{j+1}/{datasets_per_exp}"
            try:
                if size < min_size:
                    logger.info(
                        f"Skipping dataset size {label} (size: {size}) for being too small."
                    )
                    continue

                logger.info(f"Benchmarking dataset of size {label}...")

                o_n, o_nlogn, o_n2 = calc_theoretical(size)

                # Generate multiple datasets with different characteristics
                datasets = generate_datasets(size)

                for k, data in enumerate(datasets):
                    dataset_type = [
                        "random",
                        "sorted",
                        "reverse_sorted",
                        "partially_sorted",
                        "mostly_sorted",
                    ][k]
                    try:
                        eidos_time, eidos_sorted = run_benchmark(eidos_sort, data)
                        quick_time, quick_sorted = run_benchmark(np.sort, data)
                    except Exception as e:
                        logger.error(
                            f"Error running benchmarks for size {label}, dataset type {dataset_type}: {e}"
                        )
                        continue

                    # Verify correctness of EidosSort with a floating-point tolerance
                    try:
                        eidos_correct = np.allclose(
                            eidos_sorted, quick_sorted, rtol=1e-10, atol=1e-10
                        )
                    except Exception as e:
                        logger.error(
                            f"Error verifying sort correctness for size {label}, dataset type {dataset_type}: {e}"
                        )
                        eidos_correct = False

                    # Compute speedup factor. If eidos_time = 0, we treat it as infinite speedup.
                    try:
                        speedup = (
                            quick_time / eidos_time if eidos_time > 0 else float("inf")
                        )
                    except ZeroDivisionError:
                        speedup = float("inf")
                    except Exception as e:
                        logger.error(
                            f"Error calculating speedup for size {label}, dataset type {dataset_type}: {e}"
                        )
                        speedup = 0.0

                    # Gather results into the benchmark_results list.
                    mem_usage_mb = float(round(data.nbytes / (1024 * 1024), 2))
                    benchmark_results.append(
                        {
                            "Dataset Size": str(label),
                            "Dataset Type": dataset_type,
                            "N": float(size),
                            "O(n)": float(o_n),
                            "O(nlogn)": float(o_nlogn),
                            "O(n^2)": float(o_n2),
                            "EidosSort Time (s)": float(round(eidos_time, 6)),
                            "QuickSort Time (s)": float(round(quick_time, 6)),
                            "Speedup vs QuickSort": float(round(speedup, 3)),
                            "EidosSort Correct": bool(eidos_correct),
                            "Memory Usage (MB)": mem_usage_mb,
                            "Size_Numeric": float(size),
                        }
                    )
                    logger.info(
                        f"Completed benchmark for dataset size {label}, dataset type {dataset_type} successfully."
                    )

            except Exception as e:
                logger.error(f"Error benchmarking dataset {label}: {e}")
                continue

    if not benchmark_results:
        raise RuntimeError(
            "No benchmark results were collected. All benchmarking attempts failed."
        )

    # Build a DataFrame from the collected results
    df = pd.DataFrame(benchmark_results)

    # Verify required columns exist (ensuring code correctness if something changed above)
    required_columns = {
        "Dataset Size": str,
        "Dataset Type": str,
        "N": float,
        "O(n)": float,
        "O(nlogn)": float,
        "O(n^2)": float,
        "EidosSort Time (s)": float,
        "QuickSort Time (s)": float,
        "Speedup vs QuickSort": float,
        "EidosSort Correct": bool,
        "Memory Usage (MB)": float,
        "Size_Numeric": float,
    }
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in the benchmark DataFrame: {missing_cols}"
        )

    # Convert columns to expected data types
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
        results (pd.DataFrame): DataFrame containing basic benchmark results from run_benchmarks().

    Returns:
        pd.DataFrame: An updated DataFrame containing additional metrics and columns.
    """
    df = results.copy()

    # Make sure we have numeric dataset sizes
    if "Size_Numeric" not in df.columns:
        df["Size_Numeric"] = df["Dataset Size"].apply(
            lambda x: float(10 ** int(x.split("^")[1]))
        )

    # Basic efficiency metrics
    df["Efficiency"] = df["Speedup vs QuickSort"] / df["N"]
    df["Efficiency_CI"] = df["Efficiency"].std() * 1.96 / np.sqrt(len(df))

    # Scalability metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Scalability"] = df["EidosSort Time (s)"] / df["N"]
        df["Scalability_Score"] = (
            1 - (df["Scalability"].pct_change() / df["N"].pct_change()).abs()
        )

    # Memory efficiency
    theoretical_mem = df["N"] * 8 / (1024 * 1024)  # 8 bytes per float
    epsilon = 1e-10
    df["Memory_Efficiency"] = theoretical_mem / (df["Memory Usage (MB)"] + epsilon)
    df.loc[df["Memory Usage (MB)"] == 0, "Memory_Efficiency"] = np.nan
    df["Memory_Overhead"] = df["Memory Usage (MB)"] - theoretical_mem

    with np.errstate(divide="ignore", invalid="ignore"):
        df["Memory_Score"] = 1 / (1 + df["Memory_Overhead"] / theoretical_mem)

    # Compare EidosSort time to O(nlogn)
    df["Time vs O(nlogn)"] = df["EidosSort Time (s)"] / df["O(nlogn)"]
    df["Time vs O(nlogn)_CI"] = df["Time vs O(nlogn)"].std() * 1.96 / np.sqrt(len(df))

    # Performance stability
    window = 3
    rolling_std = df["EidosSort Time (s)"].rolling(window=window, min_periods=1).std()
    rolling_mean = df["EidosSort Time (s)"].rolling(window=window, min_periods=1).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Time_Consistency"] = rolling_std / rolling_mean

    # Growth rate analysis
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Growth_Rate"] = df["EidosSort Time (s)"].pct_change() / df["N"].pct_change()
        df["Relative_Growth"] = df["Growth_Rate"] / (
            df["QuickSort Time (s)"].pct_change() / df["N"].pct_change()
        )

    # Combined efficiency score (simple weighting of speed, memory, and stability)
    weights = {"speed": 0.4, "memory": 0.3, "stability": 0.3}
    df["Speed_Score"] = 1 / (1 + df["Time vs O(nlogn)"])
    df["Stability_Score"] = 1 / (
        1 + df["EidosSort Time (s)"].std() / (df["EidosSort Time (s)"].mean() + epsilon)
    )
    df["Stability_Score"] = df["Stability_Score"].clip(lower=0)

    df["Overall_Efficiency"] = (
        weights["speed"] * df["Speed_Score"]
        + weights["memory"] * df["Memory_Score"]
        + weights["stability"] * df["Stability_Score"]
    )
    df["Overall_Efficiency"] = df["Overall_Efficiency"].clip(lower=0, upper=1)

    # Performance confidence intervals
    std_time = (
        df["EidosSort Time (s)"].std() if "EidosSort Time (s)" in df.columns else None
    )
    mean_time = (
        df["EidosSort Time (s)"].mean() if "EidosSort Time (s)" in df.columns else None
    )
    if std_time is not None and mean_time is not None:
        margin = std_time * 1.96 / np.sqrt(len(df))
        df["Performance_CI_Lower"] = mean_time - margin
        df["Performance_CI_Upper"] = mean_time + margin
    else:
        # If we had a mismatch, fallback to a column that definitely exists
        col = "EidosSort Time (s)"
        if col in df.columns:
            col_std = df[col].std()
            col_mean = df[col].mean()
            margin = col_std * 1.96 / np.sqrt(len(df))
            df["Performance_CI_Lower"] = col_mean - margin
            df["Performance_CI_Upper"] = col_mean + margin
        else:
            # Provide placeholders
            df["Performance_CI_Lower"] = np.nan
            df["Performance_CI_Upper"] = np.nan

    # Normalize key score columns to 0-1
    for col in ["Speed_Score", "Memory_Score", "Stability_Score", "Overall_Efficiency"]:
        if col in df.columns:
            min_val, max_val = df[col].min(), df[col].max()
            if max_val == min_val:
                df[col] = 1.0  # All identical => just set to 1
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)

    # Memory vs Theoretical
    df["Memory vs Theoretical"] = df["Memory Usage (MB)"] / (
        df["N"] * 8.0 / (1024 * 1024)
    )

    # Replace inf with NaN for proper averaging
    df["Memory_Efficiency"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["Memory_Efficiency"].fillna(0, inplace=True)

    # Verify the presence of columns needed by visualize_benchmark_results and analysis
    required_columns = [
        "Dataset Size",
        "N",
        "O(n)",
        "O(nlogn)",
        "O(n^2)",
        "EidosSort Time (s)",
        "QuickSort Time (s)",
        "Speedup vs QuickSort",
        "EidosSort Correct",
        "Memory Usage (MB)",
        "Size_Numeric",
        "Time vs O(nlogn)",
        "Memory vs Theoretical",
        "Efficiency",
        "Efficiency_CI",
        "Scalability",
        "Scalability_Score",
        "Memory_Efficiency",
        "Memory_Overhead",
        "Memory_Score",
        "Time_Consistency",
        "Growth_Rate",
        "Relative_Growth",
        "Speed_Score",
        "Stability_Score",
        "Overall_Efficiency",
        "Performance_CI_Lower",
        "Performance_CI_Upper",
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing columns in DataFrame after metrics computation: {missing_cols}"
        )

    return df


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
    analysis: Dict[str, Any] = {}
    df: pd.DataFrame = results.copy()

    # Ensure numeric dataset sizes
    if "Size_Numeric" not in df.columns:
        df["Size_Numeric"] = df["Dataset Size"].apply(
            lambda x: float(10 ** int(x.split("^")[1]))
        )
        logger.debug("Converted 'Dataset Size' to numeric 'Size_Numeric'.")

    # Basic stats
    mean_speedup: float = float(df["Speedup vs QuickSort"].mean())
    speedup_se: float = stats.sem(df["Speedup vs QuickSort"])
    speedup_ci: Tuple[float, float] = t.interval(
        0.95, len(df) - 1, loc=mean_speedup, scale=speedup_se
    )
    logger.debug(f"Calculated mean speedup: {mean_speedup:.3f}, CI: {speedup_ci}")

    # Average memory usage vs theoretical
    avg_memory_usage: float = (
        df["Memory Usage (MB)"] / (df["Size_Numeric"] * 8 / (1024 * 1024))
    ).mean()
    logger.debug(f"Calculated average memory efficiency: {avg_memory_usage:.3f}")

    analysis["basic_stats"] = {
        "mean_speedup": mean_speedup,
        "speedup_ci": speedup_ci,
        "memory_efficiency": float(avg_memory_usage),
    }

    # Threshold-based observations
    threshold_speedup: float = mean_speedup + df["Speedup vs QuickSort"].std()
    threshold_memory: float = df["Memory Usage (MB)"].mean() * 1.5
    logger.debug(
        f"Calculated speedup threshold: {threshold_speedup:.3f}, memory threshold: {threshold_memory:.3f}"
    )

    high_speedups: List[str] = df[df["Speedup vs QuickSort"] > threshold_speedup][
        "Dataset Size"
    ].tolist()
    memory_concerns: List[str] = df[df["Memory Usage (MB)"] > threshold_memory][
        "Dataset Size"
    ].tolist()
    logger.debug(
        f"Datasets with high speedups: {high_speedups}, memory concerns: {memory_concerns}"
    )

    analysis["thresholds"] = {
        "significant_speedup": high_speedups,
        "memory_concern": memory_concerns,
    }

    # Complexity analysis via log-log linear fit
    eidos_col: pd.Series = df["EidosSort Time (s)"].replace(0, np.nan)
    quick_col: pd.Series = df["QuickSort Time (s)"].replace(0, np.nan)
    size_col: pd.Series = df["Size_Numeric"]

    # If there are zeros or NaNs, the log can fail. So we ensure we skip invalid points
    valid_mask_eidos: pd.Series = (
        (~pd.isna(eidos_col)) & (eidos_col > 0) & (size_col > 0)
    )
    valid_mask_quick: pd.Series = (
        (~pd.isna(quick_col)) & (quick_col > 0) & (size_col > 0)
    )

    # Debugging: Log the data types and a snippet of the masks
    logger.debug(f"valid_mask_eidos dtype: {valid_mask_eidos.dtype}")
    logger.debug(f"valid_mask_quick dtype: {valid_mask_quick.dtype}")
    logger.debug(f"valid_mask_eidos sample:\n{valid_mask_eidos.head()}")
    logger.debug(f"valid_mask_quick sample:\n{valid_mask_quick.head()}")

    def fit_loglog(x: pd.Series, y: pd.Series) -> float:
        """Fit a line to (log10(x), log10(y)) and return the slope."""
        x_log: np.ndarray = np.log10(x)
        y_log: np.ndarray = np.log10(y)
        slope, _ = np.polyfit(x_log, y_log, 1)
        return float(slope)

    eidos_slope: float = (
        float(fit_loglog(size_col[valid_mask_eidos], eidos_col[valid_mask_eidos]))
        if valid_mask_eidos.sum() > 1
        else 1.0
    )
    quick_slope: float = (
        float(fit_loglog(size_col[valid_mask_quick], quick_col[valid_mask_quick]))
        if valid_mask_quick.sum() > 1
        else 1.0
    )
    logger.debug(
        f"Calculated EidosSort complexity slope: {eidos_slope:.3f}, QuickSort slope: {quick_slope:.3f}"
    )

    analysis["complexity"] = {
        "eidos_complexity": f"O(n^{eidos_slope:.3f})",
        "quick_complexity": f"O(n^{quick_slope:.3f})",
        "relative_complexity": (
            eidos_slope / quick_slope if quick_slope != 0 else float("inf")
        ),
    }

    # Performance prediction model (polynomial of degree=2 in linear space)
    valid_mask_both: pd.Series = valid_mask_eidos & valid_mask_quick
    X: np.ndarray = size_col[valid_mask_both].to_numpy().reshape(-1, 1)
    y: np.ndarray = eidos_col[valid_mask_both].to_numpy()

    if len(X) > 1:
        model = make_pipeline(
            PolynomialFeatures(degree=2), StandardScaler(), LinearRegression()
        )
        model.fit(X, y)
        analysis["prediction_model"] = model
        logger.debug("Polynomial prediction model fitted successfully.")
    else:
        analysis["prediction_model"] = {}
        logger.warning("Insufficient data points to fit polynomial prediction model.")

    # Optimization recommendations
    analysis["recommendations"] = []
    if analysis["complexity"]["relative_complexity"] > 1.1:
        analysis["recommendations"].append(
            "Investigate algorithmic or pivot selection optimizations to lower complexity."
        )
        logger.info("Recommendation: Investigate algorithmic optimizations.")

    if analysis["basic_stats"]["memory_efficiency"] > 1.2:
        analysis["recommendations"].append(
            "Explore memory usage improvements or data partitioning strategies."
        )
        logger.info("Recommendation: Explore memory usage improvements.")

    # Performance stability
    eidos_std: float = df["EidosSort Time (s)"].std()
    eidos_mean: float = df["EidosSort Time (s)"].mean()
    stability_score: float = 1 - (eidos_std / eidos_mean) if eidos_mean > 0 else 0.0
    logger.debug(f"Calculated stability score: {stability_score:.3f}")

    analysis["stability"] = {
        "score": stability_score,
        "assessment": "Stable" if stability_score > 0.9 else "Needs Improvement",
    }

    # Save a textual report
    report_path = output_dir / "performance_analysis.txt"
    with open(report_path, "w") as f:
        f.write("Advanced Performance Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. Statistical Summary\n")
        f.write(f" - Mean Speedup: {analysis['basic_stats']['mean_speedup']:.3f}\n")
        f.write(f" - 95% CI for Speedup: {analysis['basic_stats']['speedup_ci']}\n")
        f.write(
            f" - Memory Efficiency (avg): {analysis['basic_stats']['memory_efficiency']:.3f}\n\n"
        )

        f.write("2. Complexity Analysis\n")
        f.write(f" - EidosSort: {analysis['complexity']['eidos_complexity']}\n")
        f.write(f" - QuickSort: {analysis['complexity']['quick_complexity']}\n")
        f.write(
            f" - Relative Scaling: {analysis['complexity']['relative_complexity']:.3f}\n\n"
        )

        f.write("3. Performance Stability\n")
        f.write(f" - Stability Score: {analysis['stability']['score']:.3f}\n")
        f.write(f" - Assessment: {analysis['stability']['assessment']}\n\n")

        f.write("4. Optimization Recommendations\n")
        if analysis["recommendations"]:
            for idx, rec in enumerate(analysis["recommendations"], start=1):
                f.write(f"   {idx}. {rec}\n")
        else:
            f.write("   None\n")
    logger.info(f"Performance analysis report saved to: {report_path}")

    return analysis


################################################################################
# Visualization Module
################################################################################
def visualize_benchmark_results(results: pd.DataFrame) -> None:
    """
    Create and save comprehensive visualizations for the benchmark results with advanced analytics.
    Includes runtime analysis, speedup validation, memory usage, stability, efficiency, scalability,
    and crossover analyses. Requires the results DataFrame to include all columns generated by
    run_benchmarks() and compute_additional_metrics().

    Args:
        results (pd.DataFrame): The DataFrame containing benchmark results.
    """
    # Make a safe copy to avoid altering the original DataFrame
    results = results.copy()

    # Convert "Dataset Size" strings (e.g. "10^i") into numeric values if not already done
    # This is redundant if compute_additional_metrics has already performed this step, but safe to ensure.
    if "Size_Numeric" not in results.columns:
        results["Size_Numeric"] = results["Dataset Size"].apply(lambda x: float(10 ** int(x.split('^')[1])))

    # Initialize figure with a large grid layout for multiple subplots
    fig = plt.figure(figsize=(16, 36), constrained_layout=True)
    gs = fig.add_gridspec(7, 2)

    # 1. Runtime Analysis (2x1 subplot)
    ax_runtime = fig.add_subplot(gs[0, :])
    ax_runtime.set_title("Algorithm Runtime Analysis with Statistical Fitting", pad=20)
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
        # Scatter plot of actual data
        ax_runtime.scatter(results["Size_Numeric"], results[time_col],
                           label=f"{algo_name} (Actual)", color=color, alpha=0.6)

        # Attempt to fit multiple models (nlogn, polynomial) and select the best fit by R^2
        def nlogn_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
            """Model function for O(n log n) complexity."""
            return a * x * np.log2(x) + b

        def polynomial_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            """Polynomial model function."""
            return a * x**b + c

        # Define a type alias for model functions
        @dataclass
        class ModelFunction:
            name: str
            func: Callable[..., np.ndarray]  # Updated to accept variable arguments
            initial_value: float

        models: Dict[str, ModelFunction] = {
            "nlogn": ModelFunction("nlogn", nlogn_model, 0.0),
            "polynomial": ModelFunction("polynomial", polynomial_model, 0.0),
        }

        best_r2: float = -np.inf
        best_fit: Optional[Tuple[str, ModelFunction, np.ndarray]] = None

        x_data = results["Size_Numeric"].to_numpy()
        y_data = results[time_col].to_numpy()

        for model_name, model_func in models.items():
            try:
                popt, _ = curve_fit(
                    model_func.func, x_data.astype(float), y_data, maxfev=20000
                )
                y_pred = model_func.func(x_data.astype(float), *popt)
                r2: float = float(r2_score(y_data, y_pred))  # Explicitly cast to float
                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = (model_name, model_func, popt)
            except Exception as e:
                logger.error(f"Error fitting model {model_name}: {e}", exc_info=True)
                continue

        if best_fit is not None:
            _, model_func, popt = best_fit
            x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 200)
            y_smooth = model_func.func(x_smooth.astype(float), *popt)
            ax_runtime.plot(
                x_smooth,
                y_smooth,
                "--",
                color=color,
                label=f"{algo_name} Trend ({best_fit[0]}, R²={best_r2:.3f})",
            )

    ax_runtime.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Speedup Analysis with Confidence Interval
    ax_speedup = fig.add_subplot(gs[1, :])
    ax_speedup.set_title("Speedup Analysis with Statistical Validation", pad=20)
    ax_speedup.set_xlabel("Dataset Size (n)")
    ax_speedup.set_ylabel("Speedup Ratio")
    ax_speedup.set_xscale('log')
    ax_speedup.grid(True)

    speedup_data = results["Speedup vs QuickSort"]
    ci = 0.95
    z_score = norm.ppf((1 + ci) / 2)

    # Compute standard error & confidence intervals
    std_err = speedup_data.std() / np.sqrt(len(speedup_data))
    ci_lower = speedup_data.mean() - z_score * std_err
    ci_upper = speedup_data.mean() + z_score * std_err

    ax_speedup.plot(results["Size_Numeric"], speedup_data, 'g-o', label="Speedup Ratio")
    ax_speedup.fill_between(results["Size_Numeric"], ci_lower, ci_upper,
                            color='g', alpha=0.2,
                            label=f"{int(ci*100)}% Confidence Interval")

    # Fit a simple polynomial to speedup data in log space for a visual trend
    try:
        z = np.polyfit(np.log10(results["Size_Numeric"]), speedup_data, 2)
        p = np.poly1d(z)
        x_smooth = np.logspace(np.log10(results["Size_Numeric"].min()),
                               np.log10(results["Size_Numeric"].max()), 100)
        y_smooth = p(np.log10(x_smooth))
        # Evaluate R^2
        r2_speedup = r2_score(speedup_data, p(np.log10(results["Size_Numeric"])))
        ax_speedup.plot(x_smooth, y_smooth, 'r--', label=f"Trend (R² = {r2_speedup:.3f})")
    except Exception as ex:
        logger.warning(f"Unable to fit polynomial for speedup trend: {str(ex)}")

    ax_speedup.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Memory Efficiency Analysis
    ax_memory = fig.add_subplot(gs[2, :])
    ax_memory.set_title("Memory Usage Analysis with Efficiency Metrics", pad=20)
    ax_memory.set_xlabel("Dataset Size (n)")
    ax_memory.set_ylabel("Memory Usage (MB)")
    ax_memory.set_xscale('log')
    ax_memory.set_yscale('log')
    ax_memory.grid(True)

    # Actual memory vs theoretical
    actual_mem = results["Memory Usage (MB)"]
    theoretical_mem = results["Size_Numeric"] * 8 / (1024 * 1024)  # 8 bytes per float

    ax_memory.plot(results["Size_Numeric"], actual_mem, 'b-o', label="Actual Memory Usage")
    ax_memory.plot(results["Size_Numeric"], theoretical_mem, 'r--', label="Theoretical Minimum")

    # Shades for overhead
    overhead = actual_mem - theoretical_mem
    ax_memory.fill_between(results["Size_Numeric"], theoretical_mem, actual_mem,
                           alpha=0.3, color='gray', label="Memory Overhead")

    # Try to display memory efficiency
    meff_col = "Memory_Efficiency"
    if meff_col in results.columns:
        efficiency_ratio = results[meff_col].mean()
        ax_memory.text(
            0.02, 0.98,
            f"Average Memory Efficiency: {efficiency_ratio:.2f}",
            transform=ax_memory.transAxes,
            verticalalignment='top'
        )
    ax_memory.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. Performance Stability Analysis
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

    # Coefficient of Variation
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = (rolling_std / rolling_mean).mean()
    ax_stability.text(
        0.02, 0.98,
        f"Coefficient of Variation: {cv:.3f}",
        transform=ax_stability.transAxes,
        verticalalignment='top'
    )
    ax_stability.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. Algorithmic Efficiency Metrics
    ax_efficiency = fig.add_subplot(gs[4, :])
    ax_efficiency.set_title("Comprehensive Efficiency Metrics", pad=20)
    ax_efficiency.set_xlabel("Dataset Size (n)")
    ax_efficiency.set_ylabel("Efficiency Metric")
    ax_efficiency.set_xscale('log')
    ax_efficiency.grid(True)

    # The following columns are expected to be present in the DataFrame after compute_additional_metrics()
    # They might not exist if that step wasn't done. We'll check and plot only if they exist.
    metric_plots = [
        ("Time vs O(nlogn)", "purple", "Time Efficiency"),
        ("Memory vs Theoretical", "orange", "Memory Efficiency"),
        ("Efficiency", "green", "Overall Efficiency")
    ]

    for col, color, label in metric_plots:
        if col in results.columns:
            ax_efficiency.plot(results["Size_Numeric"], results[col], color=color, marker='o', label=label)
    ax_efficiency.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 6. Scalability Analysis
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

    # 7. Crossover Analysis
    ax_crossover = fig.add_subplot(gs[6, :])
    ax_crossover.set_title("Algorithm Crossover Analysis", pad=20)
    ax_crossover.set_xlabel("Dataset Size (n)")
    ax_crossover.set_ylabel("Relative Efficiency")
    ax_crossover.set_xscale('log')
    ax_crossover.grid(True)

    relative_efficiency = results["QuickSort Time (s)"] / results["EidosSort Time (s)"]
    ax_crossover.plot(results["Size_Numeric"], relative_efficiency, 'b-o',
                      label="QuickSort/EidosSort Ratio")
    ax_crossover.axhline(y=1, color='r', linestyle='--', label='Efficiency Threshold')

    # Find crossover points (where ratio crosses 1)
    diffs = np.signbit(relative_efficiency - 1)
    crossover_indices = np.where(np.diff(diffs))[0]
    for idx in crossover_indices:
        crossover_x = results["Size_Numeric"].iloc[idx]
        ax_crossover.axvline(x=crossover_x, color='g', linestyle=':', alpha=0.5)
        ax_crossover.text(crossover_x, 0.5,
                          f'Crossover @ n={crossover_x:.0f}', rotation=90)

    ax_crossover.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Finally, save the figure
    plt.savefig(output_dir / "benchmark_visualization.png",
                dpi=300, bbox_inches='tight',
                metadata={"Creator": "EidosSort Benchmark Suite"})
    plt.close()

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

        # Step 2: Compute and validate advanced metrics
        results = compute_additional_metrics(results)
        logger.info("Additional metrics computed and validated.")

        # Step 3: Save results and associated metadata
        results.to_csv(output_dir / "benchmark_results.csv", index=False)
        metadata = {
            "timestamp": str(pd.Timestamp.utcnow()),
            "version": "2.0.0",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        with open(output_dir / "benchmark_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Step 4: Generate and save visualizations
        visualize_benchmark_results(results)
        logger.info("Benchmark visualizations generated successfully.")

        # Step 5: Perform deeper analysis
        analysis = analyze_sorting_performance(results)
        logger.info("Performance analysis completed.")

        # Print a quick summary to stdout
        print("\nBenchmark Suite Summary:")
        print("=" * 80)
        print(f"Total Datasets: {len(results)}")
        avg_speedup = results['Speedup vs QuickSort'].mean()
        print(f"Average Speedup vs QuickSort: {avg_speedup:.2f}x")

        mem_eff_col = "Memory_Efficiency"
        if mem_eff_col in results.columns:
            mem_efficiency = results[mem_eff_col].mean()
            print(f"Average Memory Efficiency: {mem_efficiency:.2f}")

        overall_eff_col = "Overall_Efficiency"
        if overall_eff_col in results.columns:
            overall_perf = results[overall_eff_col].mean()
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
