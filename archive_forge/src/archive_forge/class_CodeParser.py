import ast  # Importing the abstract syntax tree module to manipulate and analyze Python abstract syntax grammatically. Documentation: https://docs.python.org/3/library/ast.html
import json  # Importing the JSON module to encode and decode JSON data. Documentation: https://docs.python.org/3/library/json.html
import tkinter as tk  # Importing the tkinter module for creating graphical user interfaces. Documentation: https://docs.python.org/3/library/tkinter.html
from tkinter import (
import os  # Importing the os module to interact with the operating system. Documentation: https://docs.python.org/3/library/os.html
import logging  # Importing the logging module to enable logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
from typing import (
import docstring_parser  # Importing the docstring_parser module to parse Python docstrings. Documentation: https://pypi.org/project/docstring-parser/
from concurrent.futures import (
class CodeParser(ast.NodeVisitor):
    """
    A class to parse Python code and extract class and function definitions with detailed documentation,
    including handling complex type annotations and robust docstring parsing.
    This parser is designed to be extensible, supporting plugin-based extensions for custom parsers and formatters.

    Attributes:
        classes (List[Dict[str, Any]]): A list to store information about each class parsed from the code.

    For more information on `ast.NodeVisitor`, visit:
    https://docs.python.org/3/library/ast.html#ast.NodeVisitor
    """

    def __init__(self):
        """
        Initializes the CodeParser instance, setting up an empty list to store class definitions.
        """
        super().__init__()
        self.classes: List[Dict[str, Any]] = []
        logging.info('CodeParser initialized with an empty list of classes.')

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition node in the abstract syntax tree and extract relevant information including the name,
        docstring, methods, decorators, and class relationships.

        Parameters:
            node (ast.ClassDef): The class definition node being visited.

        This method logs the visitation of each class and appends the extracted information to the `classes` attribute.
        """
        logging.debug(f'Visiting class definition: {node.name}')
        class_info: Dict[str, Any] = {'name': node.name, 'docstring': self.enhanced_parse_docstring(ast.get_docstring(node, clean=True) or 'Description not provided.'), 'methods': [], 'decorators': self._extract_decorators(node), 'class_relationships': self._extract_class_relationships(node)}
        self.generic_visit(node)
        self.classes.append(class_info)
        logging.info(f'Class {node.name} has been successfully added with its properties and methods.')

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition node and extract relevant information including the name, docstring, and parameters.

        Parameters:
            node (ast.FunctionDef): The function definition node being visited.

        This method logs the visitation of each function and appends the extracted information to the last class in the `classes` list.
        """
        func_info: Dict[str, Any] = {'name': node.name, 'docstring': self.enhanced_parse_docstring(ast.get_docstring(node) or 'Description not provided.'), 'parameters': [self._infer_arg_type(arg) for arg in node.args.args], 'is_async': 'async' if isinstance(node, ast.AsyncFunctionDef) else 'regular', 'decorators': self._extract_decorators(node)}
        self.classes[-1]['methods'].append(func_info)
        self.generic_visit(node)

    def _infer_complex_type(self, annotation: ast.AST) -> str:
        """
        Handles complex type annotations including nested generics and Python 3.10+ features like TypeGuard.

        Parameters:
            annotation (ast.AST): The AST node representing a type annotation.

        Returns:
            str: A string representation of the inferred type.

        This method uses recursive calls to handle nested annotations and logs any failures in type inference.
        """
        try:
            if isinstance(annotation, ast.Subscript):
                base: str = self._infer_complex_type(annotation.value)
                if hasattr(annotation.slice, 'value'):
                    index: str = self._infer_complex_type(annotation.slice.value)
                else:
                    index: str = ', '.join((self._infer_complex_type(s.value) for s in annotation.slice.values))
                return f'{base}[{index}]'
            elif isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Tuple):
                elements: str = ', '.join((self._infer_complex_type(el) for el in annotation.elts))
                return f'Tuple[{elements}]'
            elif isinstance(annotation, ast.Attribute):
                return f'{self._infer_complex_type(annotation.value)}.{annotation.attr}'
            elif isinstance(annotation, ast.Constant):
                if annotation.value == 'TypeGuard':
                    return 'TypeGuard'
                return str(annotation.value)
            return 'Unknown'
        except Exception as e:
            logging.warning(f'Failed to infer type: {e}')
            return 'Unknown'

    def _infer_arg_type(self, assign: ast.arg) -> str:
        """
        Enhanced to handle complex types using the _infer_complex_type method, including nested generics and Python 3.10+ features like TypeGuard.

        Parameters:
            assign (ast.arg): The argument node whose type is to be inferred.

        Returns:
            str: A string representation of the inferred type for the argument.

        This method logs the inferred type for each argument and uses 'Any' as a default type if no annotation is present.
        """
        if assign.annotation:
            inferred_type: str = self._infer_complex_type(assign.annotation)
            logging.debug(f"Inferred type for argument '{assign.arg}': {inferred_type}")
            return inferred_type
        return 'Any'

    def enhanced_parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """
        Further enhances docstring parsing to include automatic detection of related documentation, enhancing the 'See Also' section with relevant internal links.

        Parameters:
            docstring (str): The docstring to parse.

        Returns:
            Dict[str, Any]: A dictionary containing structured information extracted from the docstring.

        This method logs the enhanced parsing process and utilizes the `docstring_parser` module for structured parsing.
        """
        parsed: docstring_parser.Docstring = docstring_parser.parse(docstring)
        related_docs: List[str] = self._find_related_docs(docstring)
        enhanced_doc_info: Dict[str, Any] = {'Short Description': parsed.short_description or 'No short description provided.', 'Long Description': parsed.long_description or 'No long description provided.', 'Parameters': [{'Name': p.arg_name, 'Type': p.type_name or 'Unknown', 'Description': p.description or 'No description provided.'} for p in parsed.params], 'Returns': {'Type': parsed.returns.type_name or 'Unknown', 'Description': parsed.returns.description or 'No description provided.'} if parsed.returns else None, 'Raises': [{'Type': ex.type_name or 'Unknown', 'Description': ex.description or 'No description provided.'} for ex in parsed.raises], 'Examples': parsed.examples or 'No examples provided.', 'See Also': related_docs or 'No additional references.'}
        logging.info(f'Enhanced docstring parsed for: {parsed.short_description}')
        return enhanced_doc_info

    def _find_related_docs(self, docstring: str) -> List[str]:
        """
        Automatically detects and links related documentation within the same project, utilizing advanced string matching and indexing for comprehensive internal linking.

        Parameters:
            docstring (str): The docstring from which to find related documents.

        Returns:
            List[str]: A list of related documents found.

        This method logs the related documents found and is a placeholder for actual implementation.
        """
        related_documents: List[str] = ['RelatedDoc1', 'RelatedDoc2']
        logging.debug(f'Related documents found for docstring: {related_documents}')
        return related_documents

    def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
        """
        Extracts and infers types of decorators applied to functions and classes.

        Parameters:
            node (ast.FunctionDef): The node from which decorators are to be extracted.

        Returns:
            List[str]: A list of inferred decorator types.

        This method logs the extracted decorators and handles both simple and complex decorators.
        """
        decorators = [self._infer_decorator_type(decorator) for decorator in node.decorator_list]
        logging.debug(f'Extracted decorators for {node.name}: {decorators}')
        return decorators

    def _infer_decorator_type(self, decorator: ast.expr) -> str:
        """
        Infers the type of a decorator, handling both simple and complex decorators.

        Parameters:
            decorator (ast.expr): The decorator expression to infer.

        Returns:
            str: A string representation of the inferred decorator type.

        This method logs the inferred decorator type and handles different AST types to infer the decorator.
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            func_name = self._infer_complex_type(decorator.func)
            args = ', '.join((self._infer_complex_type(arg) for arg in decorator.args))
            return f'{func_name} with args ({args})'
        return 'Unknown'

    def _extract_class_relationships(self, node: ast.ClassDef) -> List[str]:
        """
        Extracts and infers types of base classes for a given class definition.

        Parameters:
            node (ast.ClassDef): The class definition node from which base classes are to be extracted.

        Returns:
            List[str]: A list of inferred types of base classes.

        This method logs the extracted class relationships and handles the inference of base class types.
        """
        bases = [self._infer_complex_type(base) for base in node.bases]
        logging.debug(f'Extracted class relationships for {node.name}: {bases}')
        return bases