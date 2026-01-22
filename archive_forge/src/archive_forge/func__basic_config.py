import os
import logging
def _basic_config() -> None:
    logging.basicConfig(format='[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')