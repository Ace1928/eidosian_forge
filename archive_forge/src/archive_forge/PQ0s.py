import logging


def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="image_interconversion_gui.log",
        filemode="a",
    )
