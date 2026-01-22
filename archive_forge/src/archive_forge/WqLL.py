
    async def wrapper_logic(self, func: F, is_async: bool, *args, **kwargs) -> Any:
        """
        Contains the core logic for retrying, caching, logging, validating, and monitoring the execution
        of the decorated function. This method dynamically adapts to both synchronous and asynchronous functions,
        ensuring that the execution logic is seamlessly applied regardless of the function's nature.

        The method's functionality includes:
        - Argument validation to ensure compliance with specified criteria.
        - Caching logic for efficient retrieval of function results, minimizing execution time for repeated calls.
        - Retry mechanisms to address transient failures by re-executing the function according to predefined rules.
        - Performance logging for insights into execution efficiency.

        Args:
            func (F): The function to be executed.
            is_async (bool): Indicates if the function is asynchronous.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution, either from cache or newly computed.
        """
        # Initialize performance monitoring
        start_time = asyncio.get_event_loop().time()
        cache_key = self.generate_cache_key(
            func, args, kwargs
        )  # Assuming generate_cache_key method exists

        # Ensure thread safety with asyncio.Lock for cache operations
        async with self.cache_lock:
            # Cache logic initialization
            if self.enable_caching:
                cache_key = self.cache_key_strategy(func, args, kwargs)
                cached_result = await self.attempt_cache_retrieval(cache_key)
                if cached_result is not None:
                    logging.info(f"Cache hit for {func.__name__} with key {cache_key}")
                    return cached_result
                else:
                    logging.info(f"Cache miss for {func.__name__} with key {cache_key}")

            # Retry Logic
            if self.dynamic_retry_enabled:
                retries, delay = self.dynamic_retry_strategy(Exception)

            # Initialize retry attempt counter
            attempt = 0
            while attempt <= self.retries:
                try:
                    if is_async:
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)

                    # Cache the result if caching is enabled
                    if self.enable_caching:
                        await self.update_cache(cache_key, result, is_async)
                        logging.info(
                            f"Result cached for {func.__name__} with key {cache_key}"
                        )

                    return result
                except self.retry_exceptions as e:
                    logging.warning(
                        f"Retry {attempt + 1} for {func.__name__} due to {e}"
                    )
                    if attempt < self.retries:
                        await asyncio.sleep(delay)
                    attempt += 1
                except Exception as e:
                    logging.error(f"Exception during {func.__name__} execution: {e}")
                    raise e
                finally:
                    # Performance monitoring
                    end_time = asyncio.get_event_loop().time()
                    execution_time = end_time - start_time
                    logging.info(
                        f"Execution of {func.__name__} completed in {execution_time:.2f}s"
                    )
                    await self.log_performance(func, start_time, end_time)
