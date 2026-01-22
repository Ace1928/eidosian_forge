import csv
from config import global_config
from logger import CustomLogger


class CSVExporter:
    """
    Class responsible for exporting neural network data to CSV format.
    """

    def __init__(self):
        """
        Initializes the CSVExporter with configuration and logger.
        """
        self.config = global_config.csv_export_options
        self.logger = CustomLogger("CSVExporter").get_logger()

    def export_weights(self, weights, filename="network_weights.csv"):
        """
        Exports the weights of the neural network to a CSV file.

        Parameters:
            weights (dict): The weights of the neural network to export.
            filename (str): The name of the file to export the weights to.
        """
        try:
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(
                    file,
                    delimiter=self.config["delimiter"],
                    quotechar=self.config["quotechar"],
                    quoting=csv.QUOTE_MINIMAL,
                )
                for layer, weight_matrix in weights.items():
                    writer.writerow([layer])
                    for row in weight_matrix:
                        writer.writerow(row)
            self.logger.info(f"Weights successfully exported to {filename}.")
        except Exception as e:
            self.logger.error(f"Failed to export weights to {filename}. Error: {e}")

    def export_network_structure(
        self, network_structure, filename="network_structure.csv"
    ):
        """
        Exports the structure of the neural network to a CSV file.

        Parameters:
            network_structure (dict): The structure of the neural network to export.
            filename (str): The name of the file to export the structure to.
        """
        try:
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(
                    file,
                    delimiter=self.config["delimiter"],
                    quotechar=self.config["quotechar"],
                    quoting=csv.QUOTE_MINIMAL,
                )
                for layer, info in network_structure.items():
                    writer.writerow(
                        [layer, info["Hexagons"], info["Activation_Function"]]
                    )
            self.logger.info(f"Network structure successfully exported to {filename}.")
        except Exception as e:
            self.logger.error(
                f"Failed to export network structure to {filename}. Error: {e}"
            )


# Example usage
if __name__ == "__main__":
    from neural_network import NeuralNetwork
    from fractal_generator import FractalGenerator

    # Assuming fractal_generator and neural_network are already implemented and working
    fractal_generator = FractalGenerator(global_config.base_layer_hexagons)
    neural_network_structure = fractal_generator.generate_network()
    neural_network = NeuralNetwork(neural_network_structure)

    csv_exporter = CSVExporter()
    csv_exporter.export_weights(neural_network.weights)
    csv_exporter.export_network_structure(neural_network_structure)
