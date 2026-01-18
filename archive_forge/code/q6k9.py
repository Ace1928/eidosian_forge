import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union, Optional, Tuple, List
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UniversalDataReader:
    """
    A class meticulously designed to read and process various file types through a universal interface.
    This class encapsulates methods that identify and process file content with precision and adaptability,
    ensuring optimal performance and extensibility.
    """

    def __init__(self, file_path: str):
        """
        Initialize the UniversalDataReader with a specific file path.

        :param file_path: The path to the file to be read.
        :type file_path: str
        """
        self.file_path: str = file_path
        self.file_type: str = self.identify_file_type()
        logging.info(
            f"UniversalDataReader initialized for file: {file_path} with identified type: {self.file_type}"
        )

    def identify_file_type(self) -> str:
        """
        Determine the file type by extracting and analyzing the file extension.

        :return: A string representing the file type.
        :rtype: str
        """
        _, file_extension = os.path.splitext(self.file_path)
        file_extension = file_extension.lower()
        logging.debug(f"File extension identified: {file_extension}")
        return file_extension

    def read_file(self) -> Union[str, Dict[str, Any], pd.DataFrame, ET.ElementTree]:
        """
        Read the file based on its type and return its content in an appropriate format.

        :return: The content of the file, formatted according to its type.
        :rtype: Union[str, Dict[str, Any], pd.DataFrame, ET.ElementTree]
        """
        try:
            if self.file_type == ".json":
                return self.read_json()
            elif self.file_type == ".csv":
                return self.read_csv()
            elif self.file_type == ".xml":
                return self.read_xml()
            elif self.file_type in [".txt", ".log"]:
                return self.read_text()
            else:
                logging.error(f"Unsupported file type: {self.file_type}")
                raise ValueError(f"Unsupported file type: {self.file_type}")
        except Exception as e:
            logging.error(f"Error reading file {self.file_path}: {str(e)}")
            raise

    def read_json(self) -> Dict[str, Any]:
        """
        Read a JSON file and return its content as a dictionary.

        :return: The content of the JSON file parsed into a dictionary.
        :rtype: Dict[str, Any]
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            logging.debug(f"JSON data read successfully from {self.file_path}")
            return data

    def read_csv(self) -> pd.DataFrame:
        """
        Read a CSV file and return its content as a pandas DataFrame.

        :return: The content of the CSV file parsed into a DataFrame.
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(self.file_path)
        logging.debug(f"CSV data read successfully from {self.file_path}")
        return data

    def read_xml(self) -> ET.ElementTree:
        """
        Read an XML file and return its content as an ElementTree object.

        :return: The content of the XML file parsed into an ElementTree.
        :rtype: ET.ElementTree
        """
        tree = ET.parse(self.file_path)
        logging.debug(f"XML data read successfully from {self.file_path}")
        return tree

    def read_text(self) -> str:
        """
        Read a text file and return its content as a string.

        :return: The content of the text file.
        :rtype: str
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = file.read()
            logging.debug(f"Text data read successfully from {self.file_path}")
            return data


# Example usage:
# data_reader = UniversalDataReader('path_to_your_file.extension')
# content = data_reader.read_file()
# print(content)
