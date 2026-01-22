import functools
import random
import time
def _retry_func(func, sleep_time_generator, retries, catch_extra_error_codes, found_f, status_code_from_except_f, base_class):
    counter = 0
    for sleep_time in sleep_time_generator:
        try:
            return func()
        except Exception as exc:
            counter += 1
            if counter == retries:
                raise
            if base_class and (not isinstance(exc, base_class)):
                raise
            status_code = status_code_from_except_f(exc)
            if found_f(status_code, catch_extra_error_codes):
                time.sleep(sleep_time)
            else:
                raise